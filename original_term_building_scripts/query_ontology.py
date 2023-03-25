# -*- coding: utf-8 -*-
import os
import json
import re
import faiss
import numpy as np
from smart_open import open
import argparse
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SkillRetriever(object):
    """Class for Retrieving ontology/ESCO Skills for embedded (and contextualized) EXP, EDU and LNG skills in job ads.

    Args:
        ontotermfile = full path to ontotermfile, only specify if not default ontology is used.
        ontovecfile = full path to ontovecfile, specify if we use another embedding of ontology than default
        ontoclasslabelfile = full path to classlabelfiles of ontology, only specify if not default is used.
        rebuild_index: set to true if we need to rebuild faiss index (mostly, when ontovecfile is new/has been changed)
        batchsize = batchsize for querying with faiss


    Attributes:
        indexfile: fullpath to faiss indexfile, derived from ontoname/vecs, (for reading if exists, or writing if new)
        test: set to true if testmode (results don't make sense bc we are reading in only handful of terms)
        nneighbours:  number of the nearest terms from ontology to return
        direct_matches: if to check also for exact pattern matches
        escoskills: list of terms read in from ontology
        numeric2origLabel: dictionary from numeric ESCO classlabel to pref. class label
        numeric2contextLabel: dictionary from numeric ESCO classlabel to class label used for contextualization
    """

    def __init__(
            self,
            ontotermfile=None,
            ontovecfile=None,
            ontoclasslabelfile=None,
            rebuildindex=False,
            batchsize=None,
    ):

        self.ontotermfile = (
            "skillontology/terms-ontology-de-cleaned-ap-dedup.jsonl"
            if ontotermfile is None
            else ontotermfile
        )
        self.ontovecfile = (
            "skillontology/terms-ontology-de-cleaned-ap-dedup-withContext-mnr-sts-tsdae-jobGBERT-embs.jsonl.bz2"
            if ontovecfile is None
            else ontovecfile
        )
        self.classlabelfile = (
            "skillontology/classL4up_labels.jsonl"
            if ontoclasslabelfile is None
            else ontoclasslabelfile
        )

        self.ontoname = self.get_ontoname()
        self.rebuild_index = False if rebuildindex is False else rebuildindex
        self.indexfile = os.path.join(
            os.path.dirname(self.ontotermfile), self.ontoname + ".index"
        )

        self.test = False
        self.nneighbours = 20
        self.batch_size = 500 if batchsize is None else batchsize
        self.direct_matches = False

        # load ontolgy terms, vecs and classlabels
        self.escoskills = self.readin_ontoterms()
        self.numeric2contextLabel = self.readin_classlabels_context()
        self.numeric2origLabel = self.readin_classlabels_original()
        self.build_index()
        self.sanity_check()

    def build_index(self):

        # delete if we have index in locals (memory issues)
        if "index" in locals():
            log.info("index in locals, will be deleted")
            del index

        # rebuild index should only be set to true if new variant of ontologyvecs under the same name.
        if self.rebuild_index is False and os.path.isfile(self.indexfile):
            # try:
            index_cpu = faiss.read_index(self.indexfile)
            log.info("cpu index read from file")
            log.info(f"cpu index is trained: {index_cpu.is_trained}")
            log.info(f"type of index: {type(index_cpu)}")
        else:
            res = faiss.StandardGpuResources()  # use a single GPU
            log.info("reading in ontology embeddings...")
            self.escoembs = self.readin_ontovecs()
            d = self.escoembs.shape[1]
            index_cpu = faiss.IndexFlatIP(d)
            index_cpu.add(self.escoembs)
            log.info(f"cpu index is trained: {index_cpu.is_trained}")
            log.info(f"type of index: {type(index_cpu)}")
            faiss.write_index(index_cpu, self.indexfile)
            log.info(f"wrote index to file: {self.indexfile}")

        # index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # make it to gpu index # todo: uncomment this line when running the code on the GPU machine
        index = index_cpu
        self.index = index
        log.info(f"type of index after moving to gpu: {type(self.index)}")
        log.info(f"elements in index: {self.index.ntotal}")
        assert self.index.ntotal == len(
            self.escoskills
        ), "nr skills in ontology is not equal nr vecs in index"

    def sanity_check(self):
        k = 4
        testembs = np.asarray(
            [self.index.reconstruct(a).astype(np.float32) for a in range(80, 85)]
        )

        D, I = self.index.search(testembs, k)
        log.info("sanity check: ֱis nn of vec the vec itself, checking indexes 80-84?")
        log.info(
            f"indexes\n:{I}"
        )  # shows the 4 indexes for the 5 queries (result as array)
        log.info(
            f"distances\n:{D}"
        )  # shows the 4 distances for the 5 queries  (result as array)

    def get_ontoname(self):
        """get shortname for ontology file"""
        filename = os.path.basename(self.ontotermfile)
        fn = filename.replace("terms-", "").replace("-cleaned-ap-dedup.jsonl", "")
        if "withContext" in self.ontovecfile:
            fn += "-withContext"
        elif "noContext" in self.ontovecfile:
            fn += "noContext"
        return fn

    def readin_ontoterms(self):
        """read in ontology terms from file, return as list"""
        escoskills = []
        c = 0
        with open(self.ontotermfile, "r") as infile:
            for line in infile:
                c += 1
                if self.test == True and c == 100:
                    break
                d = json.loads(line.strip())
                escoskills.append(d)
        log.info(f"nr. ontology skills: {len(escoskills)}")
        log.debug(f"first 5 onto skills: {escoskills[0:5]}")
        return escoskills

    def readin_ontovecs(self):
        "read in ontology vectors from file, return as list"
        escoembs = []
        c = 0
        with open(self.ontovecfile, "r") as infile:
            for line in infile:
                c += 1
                if self.test == True and c == 100:
                    break
                l = json.loads(line.strip())
                escoembs.append(l)
        escoembs = np.asarray(escoembs, dtype=np.float32)
        log.info(f"shape ontology vectors: {escoembs.shape}")
        return escoembs

    def readin_classlabels_context(self):
        with open(self.classlabelfile, "r") as f:
            numeric2contextLabel = {
                json.loads(line)["label_numeric"]: json.loads(line)["label_replaced"]
                for line in f
            }
        return numeric2contextLabel

    def readin_classlabels_original(self):
        with open(self.classlabelfile, "r") as f:
            numeric2origLabel = {
                json.loads(line)["label_numeric"]: json.loads(line)["label_original"]
                for line in f
            }
        return numeric2origLabel

    def lookup_infos(self, termentry):
        "here we collect the reelvant ontology infos for vectors (same indexs in lists)"
        source = termentry["source"]
        termOrig = termentry["termOriginal"]
        hier = str(termentry["hierarchyLevel"])
        uri = termentry["uri"]
        ttype = termentry["termType"]
        cL4up = termentry["classL4up"]
        label_context = self.numeric2contextLabel[cL4up]
        label_orig = self.numeric2origLabel[cL4up]
        return hier, uri, ttype, cL4up, termOrig, source, label_context, label_orig

    def check_direct_match(self, jobadterm, jobadtermvec):
        "look up direct matches (exact string matches between job ad and ontology skills)"

        idxs = [i for i, s in enumerate(self.escoskills) if s["term"] == jobadterm]
        escoDirectMatches = []
        if len(idxs) > 0:
            for j in idxs:
                escoskill = self.escoskills[j]
                escoemb = self.index.reconstruct(j).astype(np.float32)

                (
                    hier,
                    uri,
                    ttype,
                    class1_4,
                    termOrig,
                    source,
                    label_context,
                    label_orig,
                ) = self.lookup_infos(escoskill)
                cosine = np.dot(jobadtermvec, escoemb) / (
                        np.linalg.norm(jobadtermvec) * np.linalg.norm(escoemb)
                )
                sim = round(np.float64(cosine), 3)
                escoDirectMatches.append(
                    {
                        "termEsco": escoskill["term"],
                        "cosineSim": sim,
                        "uri": uri,
                        "classL4up": class1_4,
                        "classL4up_contextLabel": label_context,
                        "classL4up_origLabel": label_orig,
                        "termOriginal": termOrig,
                        "termType": ttype,
                        "hierarchyLevel": hier,
                        "source": source,
                    }
                )
        return escoDirectMatches

    def retrieve_skills(self, span_batches):
        """perform a similarity search plus if specified an exact pattern match and add results to dict,
        remove vector from dict, yield dict"""
        # for span in spans:
        # log.info(f"nr pans for retrieving:{len(spans)}")
        for spans in span_batches:
            vs = np.asarray([span["vec"] for span in spans], dtype=np.float32)

            D, I = self.index.search(vs, self.nneighbours)

            spans_out = []
            for j, span in enumerate(spans):
                span["escoQueryResults"] = []
                span.pop("vec")
                for i in range(self.nneighbours):
                    idx = I[j][i]
                    sim = D[j][i]
                    sim = round(np.float64(sim), 3)
                    escoterm = self.escoskills[idx]
                    (
                        hier,
                        uri,
                        ttype,
                        class1_4,
                        termOrig,
                        source,
                        label_context,
                        label_orig,
                    ) = self.lookup_infos(escoterm)

                    span["escoQueryResults"].append(
                        {
                            "termEsco": escoterm["term"],
                            "cosineSim": sim,
                            "uri": uri,
                            "classL4up": class1_4,
                            "classL4up_contextLabel": label_context,
                            "classL4up_origLabel": label_orig,
                            "termOriginal": termOrig,
                            "termType": ttype,
                            "hierarchyLevel": hier,
                            "source": source,
                        }
                    )

                if self.direct_matches is True:
                    span["escoDirectMatches"] = self.check_direct_match(
                        span["term"], vs[j]
                    )
                spans_out.append(span)
            # yield spans_out #todo: change this to yeild
            return spans_out

    def collect_batch(self, ads, batch_size):
        "generator fuction to collect batch ads"
        counter = 0
        arr = []
        for ad in ads:
            arr.append(ad)
            if len(arr) == batch_size:
                counter += 1
                res = arr.copy()
                arr = []
                # 	log.info(f"last id in batch:  {ad}")

                log.info(f"yielding batch nr {counter} with {batch_size} ads")
                yield res
        # take also leftovers (last, smaller batch)
        res = arr.copy()
        arr = []
        log.info("yielding last batch with remaining ads in file")
        yield res

    def write_batches_yield_skills(self, batches, outfile):
        "generator function, dumping accoring to params, and yielding all ads anyways"
        with open(outfile, "w", encoding="utf-8") as of:
            for batch in batches:
                for ad in batch:
                    of.write(json.dumps(ad, ensure_ascii=False) + "\n")
                    yield ad

    def process_skills(self, skills, args):

        # no performane gain through batching for the size of test set, but for larger queries
        batch_size = self.batch_size if args.batch_size is None else args.batch_size

        skills_batched = self.collect_batch(skills, batch_size)
        results = self.retrieve_skills(skills_batched)

        # outdir and outfile:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        infilename = (
            os.path.basename(args.current_termfile)
            .replace(".bz2", "")
            .replace(".jsonl", "")
        )
        suffix = "-queried2" + self.ontoname
        embname_onto = get_name_ontoembs(self.ontovecfile)
        ending = ".jsonl.bz2" if args.bz2 else ".jsonl"
        outfilename = infilename + suffix + "-" + embname_onto + ending
        print("outfile", outfilename)
        outfile = os.path.join(args.outdir, outfilename)

        # write results to file, and unbatch the ads for returnning
        results = self.write_batches_yield_skills(results, outfile)

        return results


def readin_skills_vecs(infile_terms, infile_vecs):
    "function to read in term infos and vecs for job skills from diff. fiels"
    with open(infile_terms, "r") as file_terms:
        with open(infile_vecs, "r") as file_vecs:
            terms = (json.loads(l) for l in file_terms)
            vecs = (json.loads(l) for l in file_vecs)
            for t, v in zip(terms, vecs):
                t["vec"] = v
                yield t


def get_name_ontoembs(ontovecfile):
    fn_ontoembs = os.path.basename(ontovecfile)
    name_ontoembs = fn_ontoembs.replace("-embs.jsonl.bz2", "")
    name_ontoembs = re.sub(".+Context-", "", name_ontoembs)
    return name_ontoembs


def get_name_adembs(advecfile):
    fn_adembs = os.path.basename(advecfile)
    name_adembs = fn_adembs.replace("-embs.jsonl.bz2", "")
    name_adembs = name_adembs.replace("random_sample-", "").replace(
        "challenge_sample-", ""
    )
    name_adembs = name_adembs.replace("withContext-", "").replace("noContext-", "")
    return name_adembs


def check_embeddingmodel_versions(advecfile, ontovecfile):
    "just throw a warning if it seems that ontology and ad skills were not embedded with same model. adust if naming conventions are changed."
    name_ontoembs = get_name_ontoembs(ontovecfile)
    name_adembs = get_name_adembs(advecfile)
    if name_adembs != name_ontoembs:
        log.warning(
            f"are you sure yu used the same embeddingmodel for embedding ontology and ad skills? {advecfile, ontovecfile}"
        )


def main():
    description = (f"Parser for retrieving ontology-skills, via cosine embedding similarity, for embedded, "
                   f"contextualized skill areas (EDU, EXP, LNG)")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--modelpath",
        dest="adterms",
        help="path to the model checkpoint from which to create the embeddings for the job",
    )
    parser.add_argument(
        "--adterms",
        nargs="+",
        dest="adterms",
        help="path to one or more input-files with job ad-terms (as marked by NER) to process",
    )
    parser.add_argument(
        "--advecs",
        nargs="+",
        dest="advecs",
        help="path to one or more input-files with embedding vectors for ad-terms (marked by NER) to process",
    )
    parser.add_argument(
        "-l", "--logfile", dest="logfile", help="write log to FILE", metavar="FILE"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=3,
        type=int,
        metavar="LEVEL",
        help="set verbosity level: 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO 4=DEBUG (default %(default)s)",
    )
    parser.add_argument(
        "-o", "--outdir", dest="outdir", help="output directory where files are written to"
    )
    parser.add_argument(
        "--uncompressed",
        dest="bz2",
        default=True,
        action="store_false",
        help="to write files without compression",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=500,
        type=int,
        help="number of ads per batch for skill retrieving from the ontology terms (default %(default)d)",
    )
    parser.add_argument(
        "--ontoterms",
        dest="ontoterms",
        default=None,
        help=(f"path to ontology terms, if override default ontology (indicate correponding vecs too) "
              f"(default %(default)s)"),
    )
    parser.add_argument(
        "--ontovecs",
        dest="ontovecs",
        default=None,
        help=(f"path to ontology vectors, if override default ontology (indicate correponding terms too) "
              f"(default %(default)s)"),
    )
    parser.add_argument(
        "--rebuild-index",
        dest="rebuild_index",
        default=False,
        action="store_true",
        help=(f"if faiss index should be rebuild (if ontology vector file is created by a different fine-tuned model) "
              f"(default %(default)s)"),
    )
    parser.add_argument(
        "--ontoclasslabels",
        dest="ontoclasslabels",
        default=None, # todo: give a default here
        help="path to file with ontology class-labels if override default (default %(default)s)",
    )
    parser.add_argument(
        "--report-every",
        dest="report_every",
        default=10,
        type=int,
        help="log progress after every x. skill (default %(default)s)",
    )

    args = parser.parse_args()

    log_levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]

    if args.logfile:
        logging.basicConfig(filename=args.logfile, level=log_levels[args.verbose])
    else:
        logging.basicConfig(
            level=log_levels[args.verbose],
            format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
        )

    log.info(f"args: {args}")

    skillRetriever = SkillRetriever(
        ontotermfile=args.ontoterms,
        ontovecfile=args.ontovecs,
        ontoclasslabelfile=args.ontoclasslabels,
        rebuildindex=args.rebuild_index,
        batchsize=args.batch_size,
    )

    # todo:temporary additions to check the code part, fetch via a json config later
    adtermfile = 'jobads/random_sample.jsonl'
    advecfile = 'jobads_vecs/challenge_sample-withContext-mnr-sts-tsdae-jobGBERT-embs.jsonl.bz2'
    args.outdir = 'test_query_onto'
    for i in range(1):
        # with open(i, 'r', encoding='utf-8') as infile:
        log.info(f"processing infiles: {adtermfile, advecfile}")
        # skills = (json.loads(l) for l in infile)
        # args.inputfile = i
        args.current_termfile = adtermfile
        args.current_vecfile = advecfile
        check_embeddingmodel_versions(advecfile, skillRetriever.ontovecfile)
        skills = readin_skills_vecs(args.current_termfile, args.current_vecfile)
        skills = skillRetriever.process_skills(skills, args)
        c = 0
        for skill in skills:
            c += 1
            if c % args.report_every == 0:
                log.info(f"skills processed in file: {c}")


if __name__ == "__main__":
    main()
