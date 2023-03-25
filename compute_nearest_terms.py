import argparse
import bz2
import copy
import json
import logging
import os
import sys

import faiss
import numpy as np
from smart_open import open

CURRENT_FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR = 'new_results_nearest_terms'
VECTORS_DIR = 'vectors'
VECTOR_FORMAT = '.jsonl.bz2'

description = (f'File to query the random sample and the challenge terms inside the ontology to get the most similar'
               f'terms from the ontology to them')

parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "--language",
    dest="language",
    required=True,
    help=f'Specify the language for which to get the most similar terms from the ontology'
)

parser.add_argument(
    "--embedding",
    dest="embedding",
    required=True,
    help=f'Specify the directory of the vectors from where to load the index and the embeddings'
)

parser.add_argument(
    "--sim_terms",
    dest='sim_terms',
    required=True,
    help=f'The number of top most similar terms to look from inside the ontology'
)

parser.add_argument(
    "--outdir",
    dest='outdir',
    required=True,
    help=f'The path for the output directory where to store the results of the most similar terms found'
)

args = parser.parse_args()

language = args.language
embedding_dir = args.embedding
nearest_neighbours = args.sim_terms
output_dir = args.outdir

STDOUT_FILE_NAME = f'logfiles/nearest_terms_{embedding_dir}_{language}.out'
f = open(STDOUT_FILE_NAME, 'w')
sys.stdout = f

LOGFILE_NAME = f'logfiles/nearest_terms_{embedding_dir}_{language}.log'
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE_NAME,
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)
LOGGER = logging.getLogger(__name__)

VECTORS_CATEGORY = f'{language}_vectors'
OUTPUT_FOLDER = f'{language}_nearest_neighbours'
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, OUTPUT_FOLDER, output_dir)
EMBEDDINGS_DIR = os.path.join(VECTORS_DIR, VECTORS_CATEGORY, embedding_dir)

LOGGER.info(f'The configuration as per the language has been created, the following variables are created')
LOGGER.info(f'VECTORS_CATEGORY: {VECTORS_CATEGORY}')
LOGGER.info(f'OUTPUT_FOLDER: {OUTPUT_FOLDER}')
LOGGER.info(f'OUTPUT_DIRECTORY: {OUTPUT_DIR}')
LOGGER.info(f'EMBEDDINGS_DIR: {EMBEDDINGS_DIR}')

RANDOM_SAMPLE_FILE_NAME = f'jobads/random_sample_{language}.jsonl'
CHALLENGE_SAMPLE_FILE_NAME = f'jobads/challenge_sample_{language}.jsonl'
CLASSLABELS_FILE_NAME = f'skillontology/classL4up_labels_{language}.jsonl'
ESCO_ONTOLOGY_TERM_FILE_NAME = f'skillontology/ontology_terms_{language}.jsonl'

RANDOM_SAMPLE_FILE_PATH = os.path.join(CURRENT_FILE_LOCATION, RANDOM_SAMPLE_FILE_NAME)
CHALLENGE_SAMPLE_FILE_PATH = os.path.join(CURRENT_FILE_LOCATION, CHALLENGE_SAMPLE_FILE_NAME)
CLASSLABELS_FILE_PATH = os.path.join(CURRENT_FILE_LOCATION, CLASSLABELS_FILE_NAME)
ESCO_ONTOLOGY_TERM_FILE_PATH = os.path.join(CURRENT_FILE_LOCATION, ESCO_ONTOLOGY_TERM_FILE_NAME)

LOGGER.info(f'The file locations as per the language has been fetched, the following variables are created')
LOGGER.info(f'RANDOM_SAMPLE_FILE_PATH: {RANDOM_SAMPLE_FILE_PATH}')
LOGGER.info(f'CHALLENGE_SAMPLE_FILE_PATH: {CHALLENGE_SAMPLE_FILE_PATH}')
LOGGER.info(f'CLASSLABELS_FILE_PATH: {CLASSLABELS_FILE_PATH}')
LOGGER.info(f'ESCO_ONTOLOGY_TERM_FILE_PATH: {ESCO_ONTOLOGY_TERM_FILE_PATH}')


class SkillRetriever:

    def __init__(self, embeddings_path, outdir_path, random_sample_file_path, challenge_sample_file_path,
                 ontology_term_file, classlabel_file_path, nearest_neighbours):
        self.embeddings_path = embeddings_path
        self.outdir_path = outdir_path
        self.random_sample_file_path = random_sample_file_path
        self.challenge_sample_file_path = challenge_sample_file_path
        self.ontology_term_file_path = ontology_term_file
        self.classlabel_file_path = classlabel_file_path
        self.nearest_neighbours = nearest_neighbours

        self.sanity_check(self.embeddings_path, self.outdir_path)

        self.random_sample = self.read_jsonl_file(self.random_sample_file_path)
        self.challenge_sample = self.read_jsonl_file(self.challenge_sample_file_path)
        self.classlabel_replaced = self.classlabels_context(self.classlabel_file_path, 'label_replaced')
        self.classlabel_original = self.classlabels_context(self.classlabel_file_path, 'label_original')

        self.ontology_terms = self.read_jsonl_file(self.ontology_term_file_path)

        self.random_sample_vec_no_context = self.read_vectors(self.embeddings_path, self.random_sample,
                                                              'random_sample-noContext')
        self.challenge_sample_vec_no_context = self.read_vectors(self.embeddings_path, self.challenge_sample,
                                                                 'challenge_sample-noContext')
        self.ontology_index_no_context = self.read_index(self.embeddings_path, 'noContext')

        self.random_sample_vec_with_context = self.read_vectors(self.embeddings_path, self.random_sample,
                                                                'random_sample-withContext')
        self.challenge_sample_vec_with_context = self.read_vectors(self.embeddings_path, self.challenge_sample,
                                                                   'challenge_sample-withContext')
        self.ontology_index_with_context = self.read_index(self.embeddings_path, 'withContext')

        self.assert_index()

        LOGGER.info(f'Starting to create files with the top {self.nearest_neighbours} most similar terms in the output'
                    f'directory {self.outdir_path}\n')

        self.retrieve_skills_from_onto(self.random_sample_vec_no_context, self.ontology_index_no_context,
                                       sample_spec='random-sample-no-context')
        self.retrieve_skills_from_onto(self.random_sample_vec_with_context, self.ontology_index_with_context,
                                       sample_spec='random-sample-with-context')

        self.retrieve_skills_from_onto(self.challenge_sample_vec_no_context, self.ontology_index_no_context,
                                       sample_spec='challenge-sample-no-context')
        self.retrieve_skills_from_onto(self.challenge_sample_vec_with_context, self.ontology_index_with_context,
                                       sample_spec='challenge-sample-with-context')

    @staticmethod
    def sanity_check(embeddings_path, outdir_path):
        assert os.path.isdir(embeddings_path), f'There is no embedding folder at the location {embeddings_path}'

        if not os.path.isdir(outdir_path):
            os.makedirs(outdir_path)

        LOGGER.info(f'The location of the embeddings directory is valid, and the output directory is also created')

    @staticmethod
    def read_jsonl_file(jsonl_file_path):
        output_data = []
        with open(jsonl_file_path) as file:
            for line in file:
                output_data.append(json.loads(line))

        return output_data

    @staticmethod
    def classlabels_context(classlabel_file_path, key_value=None):
        with open(classlabel_file_path, "r") as file:
            numeric2contextLabel = {
                json.loads(line)["label_numeric"]: json.loads(line)[key_value]
                for line in file
            }
        return numeric2contextLabel

    @staticmethod
    def read_vectors(embeddings_path, sample_file, search_parameter):

        all_files = os.listdir(embeddings_path)
        matching_file = [file for file in all_files if search_parameter in file]
        matching_file_path = os.path.join(embeddings_path, matching_file[0])

        output_data = []
        with open(matching_file_path) as file:
            for line in file:
                output_data.append(json.loads(line))

        for sample, vector in zip(sample_file, output_data):
            sample['vector'] = vector

        return sample_file

    @staticmethod
    def read_index(embeddings_path, matching_pattern):

        all_files = os.listdir(embeddings_path)
        index_files = []
        for file in all_files:
            if file.endswith('.index'):
                index_files.append(file)

        matching_file = [file for file in index_files if matching_pattern in file]
        matching_file_path = os.path.join(embeddings_path, matching_file[0])

        index = faiss.read_index(matching_file_path)
        LOGGER.info(f"type of index after moving to gpu: {type(index)}")
        LOGGER.info(f"elements in index: {index.ntotal}")

        return index

    def assert_index(self):
        assert self.ontology_index_no_context.ntotal == len(self.ontology_terms), (f'The no-context index does not have'
                                                                                   f' the same number of terms as in'
                                                                                   f' the ontology')

        assert self.ontology_index_with_context.ntotal == len(self.ontology_terms), (f'The with-context index does not '
                                                                                     f'have the same number of terms as'
                                                                                     f' in the ontology')

    def retrieve_skills_from_onto(self, input_sample, index, sample_spec):

        sample_copy = copy.deepcopy(input_sample)

        sample_vectors = np.asarray([sample['vector'] for sample in input_sample], dtype=np.float32)

        D, I = index.search(sample_vectors, self.nearest_neighbours)

        for sample_idx, sample in enumerate(sample_copy):
            sample['escoQueryResults'] = []
            sample.pop('vector')
            for i in range(self.nearest_neighbours):
                vector_idx = I[sample_idx][i]
                similarity_score = D[sample_idx][i]
                similarity_score = round(np.float64(similarity_score), 3)
                matching_term = self.ontology_terms[vector_idx]
                matching_term['similarity_score'] = similarity_score
                sample['escoQueryResults'].append(matching_term)

        LOGGER.info(f'The nearest terms with their cosine similarities have been written in copy of the input sample '
                    f'file')

        similar_terms_output_path = os.path.join(self.outdir_path, sample_spec + VECTOR_FORMAT)
        LOGGER.info(f'The path of the file where results are saved is {similar_terms_output_path}')

        if os.path.isfile(similar_terms_output_path):
            os.remove(similar_terms_output_path)

        with bz2.open(similar_terms_output_path, 'at') as zfile:
            for line in sample_copy:
                json_line = json.dumps(line)
                zfile.write(json_line + '\n')

        LOGGER.info(f'The results have been written in the file {similar_terms_output_path}\n')


skillretriever = SkillRetriever(
    embeddings_path=EMBEDDINGS_DIR,
    outdir_path=OUTPUT_DIR,
    random_sample_file_path=RANDOM_SAMPLE_FILE_PATH,
    challenge_sample_file_path=CHALLENGE_SAMPLE_FILE_PATH,
    ontology_term_file=ESCO_ONTOLOGY_TERM_FILE_PATH,
    classlabel_file_path=CLASSLABELS_FILE_PATH,
    nearest_neighbours=int(nearest_neighbours)
)

LOGGER.info(f'All the vectors for the language {language} have been saved in the directory {OUTPUT_DIR}')
