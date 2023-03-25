# -*- coding: utf-8 -*-

from smart_open import open
import bz2
import json
import os
import numpy as np
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
import torch
import time

# flair.device = torch.device('cpu')
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
TEST = False
print(f"test mode: {TEST}. if True breaks after processing 2 batches")


def collect_batch(ads, batch_size):
    "generator fuction to collect batch ads"
    counter = 0
    arr = []
    for ad in ads:
        arr.append(ad)
        if len(arr) == batch_size:
            counter += 1
            res = arr.copy()
            arr = []
            print(f"yielding batch nr {counter} with {batch_size} ads")
            yield res
    # take also leftovers (last, smaller batch)
    res = arr.copy()
    arr = []
    print("yielding last batch with remaining ads in file")
    yield res


def readin_classlabels_context(infile):
    with open(infile, 'r') as f:
        numeric2contextLabel = {json.loads(line)['label_numeric']: json.loads(line)['label_replaced'] for line in f}
    return numeric2contextLabel


def readin_classlabels_original(infile):
    with open(infile, 'r') as f:
        numeric2origLabel = {json.loads(line)['label_numeric']: json.loads(line)['label_original'] for line in f}
    return numeric2origLabel


def _make_sentences(batch, doctype, context: bool, classLabelsContext=None):
    """make flair sentences out of contextuazlized skill areas/ontologyterms: termsPrevious and termsNext go to flair
    previous/next sentence attribute. ontology classlabel to next setnecne """
    docs = []
    for doc in batch:

        if doctype == 'jobads':
            sentence = Sentence(doc['term'])
            previous = None
            if context is True and doc['termsPrevious'] != "":
                previous = doc['termsPrevious'] + ', '
            next = None
            if context is True and doc['termsNext'] != "":
                next = ', ' + doc['termsNext']

            sentence._previous_sentence = Sentence(previous) if previous is not None else None
            sentence._next_sentence = Sentence(next) if next is not None else None
            docs.append(sentence)

        elif doctype == 'ontologyterms':
            sentence = Sentence(doc['term'])

            previous = None
            next = None

            classlabel = classLabelsContext[doc['classL4up']]

            if context is True and classlabel != "" and not classlabel.isspace() and not classlabel == doc['term']:
                next = ' ( ' + classlabel + ' )'
            log.debug(f"term with context that goes into next: {doc['term'], ' - ', next}")
            sentence._previous_sentence = Sentence(previous) if previous is not None else None
            sentence._next_sentence = Sentence(next) if next is not None else None
            docs.append(sentence)
        else:
            sentence = Sentence(doc)
            if context is True:
                log.warning("context set to True, but no context given for doctype not in ['jobads', 'ontology']")
            docs.append(sentence)
    return docs


def _embed_sentences(docs, embeddings):
    """embed the whole contextualized sentence (sentence plus previous and next sentece), then extract mean of
    subtoken embeddings only for sentence of interest (without prev. and next sentence) """
    expanded_sentences = []
    context_offsets = []
    for s in docs:
        expanded_sentence, left_context_length = embeddings._expand_sentence_with_context(s)
        expanded_sentences.append(expanded_sentence)
        context_offsets.append(left_context_length)

    # tokenize
    tokenized_sentences, all_token_subtoken_lengths, subtoken_lengths = embeddings._gather_tokenized_strings(
        expanded_sentences)
    batch_encoding = embeddings.tokenizer(
        tokenized_sentences,
        stride=embeddings.stride,
        return_overflowing_tokens=embeddings.allow_long_sentences,
        truncation=embeddings.truncate,
        padding=True,
        return_tensors="pt",
    )

    input_ids, model_kwargs = embeddings._build_transformer_model_inputs(batch_encoding, tokenized_sentences, docs)

    # save sub-token offsets (bc we take later mean of embeddings only of sub-tokens of original sentence)
    context_offsets_subtokens = []
    end_originalsens_subtokens = []
    for o, st, sen in zip(context_offsets, all_token_subtoken_lengths, docs):
        start = sum([i for i in st[:o]]) + 1
        context_offsets_subtokens.append(start)
        end = sum([i for i in st[: o + len(sen)]]) + 1
        end_originalsens_subtokens.append(end)
    # print(embeddings.tokenizer.convert_ids_to_tokens(i[start:end]))
    # for i, s, e, o in zip(input_ids,context_offsets_subtokens, end_originalsens_subtokens, context_offsets):
    # 	print('******')
    # 	print(embeddings.tokenizer.convert_ids_to_tokens(i))
    # 	print(s, e, o)
    # 	print(embeddings.tokenizer.convert_ids_to_tokens(i[s:e]))

    with torch.no_grad():

        hidden_states = embeddings.model(input_ids, **model_kwargs)[-1]
        # make the tuple a tensor; makes working with it easier.
        hidden_states = torch.stack(hidden_states)
        # only use layers that will be outputted
        hidden_states = hidden_states[embeddings.layer_indexes, :, :]
        if embeddings.allow_long_sentences:
            sentence_hidden_states = embeddings._combine_strided_sentences(
                hidden_states,
                sentence_parts_lengths=torch.unique(
                    batch_encoding["overflow_to_sample_mapping"],
                    return_counts=True,
                    sorted=True,
                )[1].tolist(),
            )
        else:
            # print('befor perm.', sentence_hidden_states)
            sentence_hidden_states = list(hidden_states.permute((1, 0, 2, 3)))
        # print('after perm.', sentence_hidden_states)
        # remove padding and context (bc we only want mean of original sentence)
        sentence_hidden_states = [
            sentence_hidden_state[:, context_offsets_subtoken: end_originalsens_subtoken, :]
            for (context_offsets_subtoken, end_originalsens_subtoken, sentence_hidden_state) in
            zip(context_offsets_subtokens, end_originalsens_subtokens, sentence_hidden_states)
        ]
        # now extract
        embeddings._extract_document_embeddings(sentence_hidden_states,
                                                docs)  # , context_offsets_subtokens, end_originalsens_subtokens)
    return docs


def embed_docs(indir, infile, outdir, embeddings, emb_name, doctype='jobads', context=True, classLabelsContext=None):
    print(f"indir: {indir}, infile: {infile}")

    # set outfile name
    if context is False:
        outfile_ending = '-noContext-' + emb_name + '-embs.jsonl.bz2'
    elif context is True:
        outfile_ending = '-withContext-' + emb_name + '-embs.jsonl.bz2'

    outfile = infile.replace('.bz2', '').replace('.jsonl', '') + outfile_ending

    if TEST is True:
        outfile = 'TEST-' + outfile
    print(f"outdir: {outdir}, outfile: {outfile}")

    with open(os.path.join(indir, infile), 'rt', encoding='utf-8') as f:
        ads = (json.loads(l) for l in f)
        # too large batches are slowing down
        batches = collect_batch(ads, 100)

        with bz2.open(os.path.join(outdir, outfile), 'wt') as zfile:
            c = 0
            for b in batches:
                c += 1
                print(f"processing batch nr {c}")
                if c == 2 and TEST == True:
                    break
                # torch.cuda.empty_cache() #not helpful
                sentences = _make_sentences(b, doctype, context, classLabelsContext)
                embedded_sentences = _embed_sentences(sentences, embeddings)
                for s in embedded_sentences:
                    # to write emb, send to cpu, normalize length, to list and json dump.
                    e = s.get_embedding()
                    e = e.cpu()
                    # print('lenght before:', np.linalg.norm(e))
                    e_norm = e / np.sqrt((e ** 2).sum())
                    # print('lenght after:', np.linalg.norm(e_norm))
                    l = e_norm.tolist()
                    j = json.dumps(l)
                    zfile.write(j + '\n')


def main():
    jobads = False
    onto = True

    # two best models according to our experiments
    embedding_paths = [
        'mnr_sts_tsdae_jobGBERT',
        #	'/home/ubuntu/switchdrive/NRP77/models/SBERT/mnr_sts_jobGBERT',
    ]

    for emb_path in embedding_paths:
        emb_name = os.path.basename(emb_path).replace('_', '-')
        print(f"embedding name:  {emb_name} ")
        print(f"embedding path: {emb_path}")
        emb = TransformerWordEmbeddings(emb_path, layers='-1', layer_mean=True, subtoken_pooling='first')
        # set these here, bc. not all can be set in init
        emb.context_length = 64
        emb.respect_document_boundaries = False
        emb.cls_pooling = 'mean'
        emb.allow_long_sentences = True

        if jobads == True:

            # embedding jobads
            indir = 'jobads'
            outdir = 'jobads_vecs'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            infiles = ['challenge_sample.jsonl', 'random_sample.jsonl']

            # these below are just additional files that can be used for experimentation, now goldtandard avaialable though!
            # infiles = ['jobadterms-EDU-contextualized-sample.jsonl', 'jobadterms-EXP-contextualized-sample.jsonl']

            for infile in infiles:
                start = time.time()
                embed_docs(indir, infile, outdir, emb, emb_name, doctype='jobads', context=True,
                           classLabelsContext=None)
                end = time.time()
                print(f"took {round((end - start) / 60, 2)} minutes to embed")

        if onto == True:

            # for embedding ontologies with context, we ne need the custom made class labels for contextualization
            numeric2contextLabel = readin_classlabels_context(
                'skillontology/classL4up_labels.jsonl')
            # for k, v in numeric2contextLabel.items():
            #	print(k, v)
            indir = 'skillontology/'
            infiles = ['terms-ontology-de-cleaned-ap-dedup.jsonl']
            outdir = 'onto_vecs'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            contextualization = [True, False]
            for infile in infiles:
                for con in contextualization:
                    start = time.time()
                    embed_docs(indir, infile, outdir, emb, emb_name, doctype='ontologyterms', context=con,
                               classLabelsContext=numeric2contextLabel)
                    end = time.time()
                    print(f"took {round((end - start) / 60, 2)} minutes to embed")


if __name__ == '__main__':
    main()
