import argparse
import bz2
import copy
import json
import logging
import shutil

import faiss
import numpy as np
import os
import os
import pandas as pd
import sys
import time
import torch
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from smart_open import open
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_CACHE'] = '/cluster/scratch/yakram/conda_env/'
os.environ['HF_HOME'] = '/cluster/scratch/yakram/conda_env/'
os.environ['FLAIR_CACHE_ROOT'] = '/cluster/scratch/yakram/conda_env/'


description = (f'File to create vectors for the jobads and the ontology terms from the model checkpoint and save '
               f'them. Must be used to recreate the vectors when the base model is changed')

parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "--language",
    dest="language",
    required=True,
    help=f'Specify the path of the model to use to create the vector embeddings from jobads and onotology terms'
)

parser.add_argument(
    "--checkpoint_directory",
    dest="checkpoint_directory",
    required=True,
    help=f'Specify the directory of the model from which we need the checkpoint'
)

parser.add_argument(
    "--checkpoint_path",
    dest="checkpoint_path",
    required=True,
    help=f'Specify the path of the model to use to create the vector embeddings from jobads and onotology terms'
)

parser.add_argument(
    "--modelname",
    dest="modelname",
    required=True,
    help=f'Specify the pseudo name of the model used to create the vector embeddings from jobads and onotology '
         f'terms'
)

parser.add_argument(
    "--outdir",
    dest='outdir',
    required=True,
    help=f'The path for the output directory where to store the model in SCRATCH base path'
)

parser.add_argument(
    "--run_mode",
    dest='run_mode',
    required=True,
    help=f'The path for the output directory where to store the model in SCRATCH base path'
)
args = parser.parse_args()

language = args.language
checkpoint_dir = args.checkpoint_directory
checkpoint_path = args.checkpoint_path

# model name and output dir can be synchronized with the checkpoint dir name
model_name = args.modelname
output_dir_name = args.outdir
run_mode = args.run_mode

CURRENT_FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))
if run_mode == 'euler':
    BASE_VECTOR_OUT_PATH = '/cluster/scratch/yakram/sbert-copy/new_vectors'
elif run_mode == 'local':
    BASE_VECTOR_OUT_PATH = '/cluster/scratch/yakram/sbert-copy/new_vectors'

BASE_DATA_DIR = '/cluster/scratch/yakram'
VECTOR_FORMAT = 'embs.jsonl.bz2'
INDEX_FORMAT = 'embs.index'

STDOUT_FILE_NAME = f'logfiles/build_index_{model_name}_{checkpoint_path}_{language}.out'
f = open(STDOUT_FILE_NAME, 'w')
sys.stdout = f

LOGFILE_NAME = f'logfiles/build_index_{model_name}_{checkpoint_path}_{language}.log'
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE_NAME,
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)
LOGGER = logging.getLogger(__name__)

if language == 'en':
    VECTORS_DIRECTORY = f'{language}_vectors'
    BASE_MODEL_DIR = os.path.join(BASE_DATA_DIR, 'en_models')

    CHECKPOINT_PATH = os.path.join(BASE_MODEL_DIR, checkpoint_dir, checkpoint_path)  # todo: uncomment this later
    if checkpoint_path == 'xlm-roberta-base':
        CHECKPOINT_PATH = 'xlm-roberta-base'
    OUTPUT_DIRECTORY = os.path.join(BASE_VECTOR_OUT_PATH, VECTORS_DIRECTORY, output_dir_name)

elif language == 'de':
    VECTORS_DIRECTORY = f'{language}_vectors'
    BASE_MODEL_DIR = os.path.join(BASE_DATA_DIR, 'de_models')
    CHECKPOINT_PATH = os.path.join(BASE_MODEL_DIR, checkpoint_dir, checkpoint_path)

    if checkpoint_path == 'xlm-roberta-base':
        CHECKPOINT_PATH = 'xlm-roberta-base'

    OUTPUT_DIRECTORY = os.path.join(BASE_VECTOR_OUT_PATH, VECTORS_DIRECTORY, output_dir_name)

LOGGER.info(f'The configuration as per the language has been created, the following variables are created')
LOGGER.info(f'BASE_MODEL_DIR: {BASE_MODEL_DIR}')
LOGGER.info(f'CHECKPOINT_PATH: {CHECKPOINT_PATH}')
LOGGER.info(f'OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}')

RANDOM_SAMPLE_FILE_NAME = f'jobads/random_sample_{language}.jsonl'
CHALLENGE_SAMPLE_FILE_NAME = f'jobads/challenge_sample_{language}.jsonl'
CLASSLABELS_FILE_NAME = f'skillontology/classL4up_labels_{language}.jsonl'
ESCO_ONTOLOGY_TERM_FILE_NAME = f'skillontology/ontology_terms_{language}.jsonl'

BASE_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
RANDOM_SAMPLE_FILE_PATH = os.path.join(BASE_FILE_PATH, '../', RANDOM_SAMPLE_FILE_NAME)
CHALLENGE_SAMPLE_FILE_PATH = os.path.join(BASE_FILE_PATH, '../', CHALLENGE_SAMPLE_FILE_NAME)
CLASSLABELS_FILE_PATH = os.path.join(BASE_FILE_PATH, '../', CLASSLABELS_FILE_NAME)
ESCO_ONTOLOGY_TERM_FILE_PATH = os.path.join(BASE_FILE_PATH, '../', ESCO_ONTOLOGY_TERM_FILE_NAME)

LOGGER.info(f'The file locations as per the language has been fetched, the following variables are created')
LOGGER.info(f'BASE_FILE_PATH: {BASE_FILE_PATH}')
LOGGER.info(f'RANDOM_SAMPLE_FILE_PATH: {RANDOM_SAMPLE_FILE_PATH}')
LOGGER.info(f'CHALLENGE_SAMPLE_FILE_PATH: {CHALLENGE_SAMPLE_FILE_PATH}')
LOGGER.info(f'CLASSLABELS_FILE_PATH: {CLASSLABELS_FILE_PATH}')
LOGGER.info(f'ESCO_ONTOLOGY_TERM_FILE_PATH: {ESCO_ONTOLOGY_TERM_FILE_PATH}')


def sanity_checks(model_path, outdir_path):
    LOGGER.info(f'Performing sanity check')
    if model_path != 'xlm-roberta-base':
        assert os.path.isdir(model_path), f'There is no model checkpoint available at the mentioned path: {model_path}'

    if not os.path.isdir(outdir_path):
        os.makedirs(outdir_path)

    LOGGER.info(f'Sanity check completed')


def embed_sentences(model, sentences_list, context=False):
    expanded_sentences = []
    context_offsets = []

    # if context:
    #     LOGGER.info(f'The context is True, so the vectors for contextualized job ads are being created')
    # else:
    #     LOGGER.info(f'The context is False, so the vectors for non contextualized job ads are being created')

    for sentence in sentences_list:
        expanded_sentence, left_context_length = model._expand_sentence_with_context(sentence)
        expanded_sentences.append(expanded_sentence)
        context_offsets.append(left_context_length)

    # LOGGER.info(f'The context offsets are computed')

    # this function gives us the input sentence tokenized, alongwith the length of the tokens each token in the input
    # is subdivided into, and the length of the total number of tokens once the token is subdivided
    tokenized_sentences, len_subtoken_each, all_subtoken_lengths = model._gather_tokenized_strings(
        expanded_sentences
    )

    # this method gives us the input ids of the tokens from a map, which are padded upto the maximum length that of
    # the largest term
    tokenized_inputs_all = model.tokenizer(
        tokenized_sentences,
        stride=model.stride,
        return_overflowing_tokens=model.allow_long_sentences,
        truncation=model.truncate,
        padding=True,
        return_tensors='pt'
    )

    # this gives the input ids separately, along-with the attention mask layer and other arguments that the model needs
    input_ids, model_kwargs = model._build_transformer_model_inputs(
        tokenized_inputs_all, tokenized_sentences,
        sentences_list
    )

    context_offsets_subtokens = []
    end_original_sentence_subtokens = []

    for offsets, individual_subtoken_length, original_term in zip(
            context_offsets, len_subtoken_each, sentences_list
    ):
        # get the start of the original term when left context is added (start after termsPrevious added to orig term)

        # 1 is added since the first token in CLS token in embeddings
        original_term_start = sum([i for i in individual_subtoken_length[:offsets]]) + 1
        context_offsets_subtokens.append(original_term_start)
        end = sum([i for i in individual_subtoken_length[:offsets + len(original_term)]]) + 1  # 1 added to offset start
        end_original_sentence_subtokens.append(end)

    with torch.no_grad():
        hidden_states = model.model(input_ids, **model_kwargs)[-1]
        hidden_states = torch.stack(hidden_states)
        # only use layers that will be outputted
        hidden_states = hidden_states[model.layer_indexes, :, :]

        # this function assigns the embedding of each sentence to itself, instead of having all the embeddings in list
        if model.allow_long_sentences:
            sentence_hidden_states = model._combine_strided_sentences(
                hidden_states,
                sentence_parts_lengths=torch.unique(
                    tokenized_inputs_all["overflow_to_sample_mapping"],
                    return_counts=True,
                    sorted=True,
                )[1].tolist(),
            )

        # only select the first sentence by the subtoken length

        sentence_hidden_states_term_only = [
            sentence_hidden_state[:,
            context_offsets_subtoken: end_original_sentence_subtoken, :]
            for context_offsets_subtoken, end_original_sentence_subtoken, sentence_hidden_state in zip(
                context_offsets_subtokens, end_original_sentence_subtokens, sentence_hidden_states
            )
        ]

        model._extract_document_embeddings(sentence_hidden_states_term_only, sentences_list)

    # LOGGER.info(f'The embeddings are now attached inside the sentences list initially passed')

    return sentences_list


def create_batch(esco_terms, batch_size):
    counter = 0
    arr = []
    for term in esco_terms:
        arr.append(term)
        if len(arr) == batch_size:
            counter += 1
            res = arr.copy()
            arr = []
            LOGGER.info(f"yielding batch nr {counter} with {batch_size} ads")
            yield res

    # take also leftovers (last, smaller batch)
    res = arr.copy()
    arr = []
    print("yielding last batch with remaining ads in file")
    yield res


def create_and_store_embeddings(dataset, output_path, model, context, doctype, sample_name, numeric2contextLabel=None):
    if doctype == 'jobads':
        vector_file_suffixes = ['-withContext-', '-noContext-']
        vector_file_names = [sample_name + suffix + VECTOR_FORMAT for suffix in vector_file_suffixes]
        vector_file_paths = [os.path.join(output_path, file_name) for file_name in vector_file_names]
        for file in vector_file_paths:
            if os.path.exists(file):
                LOGGER.warning(f'The file {file} already exists, it will be overwritten to save the new vector file '
                               f'instead')
                os.remove(file)
                os.mknod(file)
        LOGGER.info(f'Collecting sentences for document type: {doctype}')
        original_sentences = []
        contextualized_sentences = []

        for line in dataset:
            line = json.loads(line)
            original_sentence = Sentence(line['term'])
            contextualized_sentence = copy.deepcopy(original_sentence)

            if line['termsPrevious'] != '':
                left_sentence = Sentence(line['termsPrevious'] + ', ')
                contextualized_sentence._previous_sentence = left_sentence
            if line['termsNext'] != '':
                right_sentence = Sentence(', ' + line['termsNext'])
                contextualized_sentence._next_sentence = right_sentence

            original_sentences.append(original_sentence)
            contextualized_sentences.append(contextualized_sentence)

        LOGGER.info(f'The contextualized and the original sentences for the sample {sample_name} have been collected')

        LOGGER.info(f'Creating embeddings for contextualized sentences for sample {sample_name} and document {doctype}')
        job_terms_contextualized = embed_sentences(model, contextualized_sentences, context)

        LOGGER.info(f'Creating embeddings for original sentences for sample {sample_name} and document {doctype}')
        job_terms_nocontextualized = embed_sentences(model, original_sentences, context)

        job_terms_all = [job_terms_contextualized, job_terms_nocontextualized]

        LOGGER.info(f'The vector file paths are: {vector_file_paths}. Embeddings will be written in these paths')

        for file_path, job_embedding in zip(vector_file_paths, job_terms_all):
            with bz2.open(file_path, 'at') as zfile:
                for sentence in job_embedding:
                    embedding = sentence.get_embedding()
                    embedding = embedding.cpu()
                    embedding_normalized = embedding / np.sqrt((embedding ** 2).sum())
                    embedding_list = embedding_normalized.tolist()
                    embedding_json = json.dumps(embedding_list)
                    zfile.write(embedding_json + '\n')

        LOGGER.info(f'The embeddings for jobads with context: {context} are written\n')

    elif doctype == 'esco':
        LOGGER.info(f'Collecting sentences for document type: {doctype}')
        vector_file_suffixes = ['-withContext-', '-noContext-']
        vector_file_names = [sample_name + suffix + VECTOR_FORMAT for suffix in vector_file_suffixes]
        vector_file_paths = [os.path.join(output_path, file_name) for file_name in vector_file_names]
        for file in vector_file_paths:
            if os.path.exists(file):
                LOGGER.warning(f'The file {file} already exists, it will be overwritten to save the new vector file '
                               f'instead')
                os.remove(file)
                os.mknod(file)
        with open(dataset, 'rt', encoding='utf-8') as file:
            esco_terms = (json.loads(line) for line in file)
            esco_terms_batch = create_batch(esco_terms, 100)

            for batch in esco_terms_batch:
                original_sentences = []
                contextualized_sentences = []
                for line in batch:
                    original_sentence = Sentence(line['term'])
                    original_sentences.append(original_sentence)

                    contextualized_sentence = copy.deepcopy(original_sentence)
                    classL4up = line['classL4up']
                    if 'http://data.europa.eu/esco/' in classL4up:
                        classL4up = line['classL4up'].replace('http://data.europa.eu/esco/', '')
                    classlabel = numeric2contextLabel[classL4up]
                    next_sentence = ' ( ' + classlabel + ' )'
                    contextualized_sentence._next_sentence = Sentence(next_sentence)
                    contextualized_sentences.append(contextualized_sentence)

                esco_terms_contextualized = embed_sentences(model, contextualized_sentences, context)
                esco_terms_nocontextualized = embed_sentences(model, original_sentences, context)

                esco_terms_all = [esco_terms_contextualized, esco_terms_nocontextualized]

                for file_path, esco_embedding in zip(vector_file_paths, esco_terms_all):
                    with bz2.open(file_path, 'at') as zfile:
                        for sentence in esco_embedding:
                            embedding = sentence.get_embedding()
                            embedding = embedding.cpu()
                            embedding_normalized = embedding / np.sqrt((embedding ** 2).sum())
                            embedding_list = embedding_normalized.tolist()
                            embedding_json = json.dumps(embedding_list)
                            zfile.write(embedding_json + '\n')

        LOGGER.info(f'The vectors for esco terms are written in paths {vector_file_paths}')

        index_file_names = [sample_name + suffix + INDEX_FORMAT for suffix in vector_file_suffixes]
        index_file_paths = [os.path.join(output_path, file_name) for file_name in index_file_names]

        LOGGER.info(f'The index paths are {index_file_paths}')

        for embedding_file_path, index_file_path in zip(vector_file_paths, index_file_paths):
            file_embeddings = []
            with open(embedding_file_path, 'r') as embedding_file:
                for line in embedding_file:
                    embedding_line = json.loads(line.strip())
                    file_embeddings.append(embedding_line)

            LOGGER.info(f'The embedding file {embedding_file_path} has been loaded')

            file_embeddings = np.asarray(file_embeddings, dtype=np.float32)
            file_dimensions = file_embeddings.shape[1]
            res = faiss.StandardGpuResources()
            index_cpu = faiss.IndexFlatIP(file_dimensions)
            index_cpu.add(file_embeddings)
            faiss.write_index(index_cpu, index_file_path)
            LOGGER.info(f'The index in the path {index_file_path} is written')

        LOGGER.info(f'All the indexes for the esco terms is now written in the directory {index_file_paths}')


def create_job_ads_embeddings(model_name, outdir_path, model):
    LOGGER.info(f'Starting the embedding workflow for job ads data')
    emb_name_with_context = model_name + '-withContext-embs.jsonl.bz2'
    emb_name_without_context = model_name + '-noContext-embs.jsonl.bz2'

    emb_with_context = os.path.join(outdir_path, emb_name_with_context)
    emb_without_context = os.path.join(outdir_path, emb_name_without_context)

    LOGGER.info(f'The embeddings "with" context will be stored in file {emb_with_context}')
    LOGGER.info(f'The embeddings "without" context will be stored in file {emb_without_context}')

    with open(RANDOM_SAMPLE_FILE_PATH, 'rt', encoding='utf-8') as file:
        random_sample = []
        for line in file:
            random_sample.append(line)

    LOGGER.info(f'The random sample file has been loaded')

    with open(CHALLENGE_SAMPLE_FILE_PATH, 'rt', encoding='utf-8') as file:
        challenge_sample = []
        for line in file:
            challenge_sample.append(line)

    LOGGER.info(f'The challenge sample file has been loaded')

    create_and_store_embeddings(random_sample, outdir_path, model, context=True, doctype='jobads',
                                sample_name='random_sample')
    create_and_store_embeddings(challenge_sample, outdir_path, model, context=False, doctype='jobads',
                                sample_name='challenge_sample')


def create_esco_embeddings(model_name, outdir_path, model):
    with open(CLASSLABELS_FILE_PATH, 'r') as file:
        numeric2contextLabel = {json.loads(line)['label_numeric']: json.loads(line)['label_replaced'] for line in file}

    create_and_store_embeddings(ESCO_ONTOLOGY_TERM_FILE_PATH, outdir_path, model, context=True, doctype='esco',
                                sample_name='esco', numeric2contextLabel=numeric2contextLabel)


def main():
    # todo: uncomment this block of code
    sanity_checks(CHECKPOINT_PATH, OUTPUT_DIRECTORY)

    model = TransformerWordEmbeddings(CHECKPOINT_PATH, layers='-1', layer_mean=True, subtoken_pooling='first',
                                      cls_pooling='mean', allow_long_sentences=True,
                                      respect_document_boundaries=False, cache_dir='/cluster/scratch/yakram/conda_env/'
                                      )

    model.context_length = 64

    if len(os.listdir(OUTPUT_DIRECTORY)) > 0:
        shutil.rmtree(OUTPUT_DIRECTORY)
        os.makedirs(OUTPUT_DIRECTORY)

    LOGGER.info(f'Creating embeddings for job ads random and challenge sample')
    create_job_ads_embeddings(model_name, OUTPUT_DIRECTORY, model)

    LOGGER.info(f'Creating embeddings for ESCO data')
    create_esco_embeddings(model_name, OUTPUT_DIRECTORY, model)

    LOGGER.info(f'The script has finished')

    f.close()


if __name__ == '__main__':
    main()
