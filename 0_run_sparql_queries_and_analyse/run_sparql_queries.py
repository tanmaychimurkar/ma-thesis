import datetime
import json
import logging
import os

import pandas as pd
import rdflib
from smart_open import open
from tqdm import tqdm


class DuplicateFileError(Exception):
    pass


CURRENT_DIRECTORY = os.path.dirname(__file__)
QUERY_FOLDER = "sparql/language_queries"
CONFIG_FILE = "language_data.json"

CONFIG_KEYS = [
    "language",
    "turtle_file",
    "logfile_name",
    "results_directory",
    "resume_queries",
    "limit_data",
    "run_mode",
]
CONFIG_FILE_PATH = os.path.join(CURRENT_DIRECTORY, CONFIG_FILE)

with open(CONFIG_FILE_PATH) as f:
    CONFIGURATION = json.load(f)

LANGUAGE = CONFIGURATION["language"]
TURTLE_FILE = CONFIGURATION["turtle_file"]
LOGFILE_NAME = CONFIGURATION["logfile_name"]
LANGUAGE_RESULTS_DIR = CONFIGURATION["results_directory"]
RUN_MODE = CONFIGURATION["run_mode"]

DATA_DIR_PATH = "/cluster/scratch/yakram/sbert_data/turtle_files"
if RUN_MODE == "local":
    DATA_DIR_PATH = "/home/tanmay/thesis/code/sbert-adaptation/turtle_files"  # todo: check this parameter always

BASE_LOG_DIR = "logfiles"
LOG_DIR = os.path.join(CURRENT_DIRECTORY, BASE_LOG_DIR, LANGUAGE)
DATE_TODAY = str(datetime.datetime.today().date())
LOGFILE_NAME_WITH_DATE = f"{LOGFILE_NAME}-{DATE_TODAY}.log"
LOGFILE_PATH = os.path.join(LOG_DIR, LOGFILE_NAME_WITH_DATE)

QUERY_FOLDER_PATH = os.path.join(CURRENT_DIRECTORY, '../', QUERY_FOLDER)
TURTLE_FILE_PATH = os.path.join(DATA_DIR_PATH, TURTLE_FILE)
LANGUAGE_RESULTS_DIR_PATH = os.path.join(DATA_DIR_PATH, LANGUAGE_RESULTS_DIR)

if not (os.path.isdir(LOG_DIR)):
    os.makedirs(LOG_DIR)

if os.path.isfile(LOGFILE_PATH):
    if RUN_MODE == 'remote':
        raise DuplicateFileError(
            f"The logfile already exists, please choose a different name for the logfile from the"
            f" language_data.json configuration"
        )
    elif RUN_MODE == 'local':
        os.remove(LOGFILE_PATH)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE_PATH,
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)


class languageDataFetcher:
    def __init__(self):
        self.esco_graph = None
        self.output_directory = LANGUAGE_RESULTS_DIR_PATH
        self.query_folder = QUERY_FOLDER_PATH
        self.esco_graph_path = TURTLE_FILE_PATH
        self.language_config = self.load_config(CONFIG_FILE_PATH)
        self.language_used = self.language_config["language"]

    @staticmethod
    def load_config(config_path):
        with open(config_path) as config:
            config_data = json.load(config)

        assert set(CONFIG_KEYS).issubset(list(config_data.keys()))
        LOGGER.info(
            f'The configuration has been loaded, the language passed is {config_data["language"]}'
        )

        LOGGER.info(f"The job is starting with the configuration")
        LOGGER.info(CONFIGURATION)
        return config_data

    def check_directories(self):
        LOGGER.info(
            f"The following directories will be checked: query folder and language results"
        )

        result_directory = self.output_directory

        if not (os.path.isdir(result_directory)):
            LOGGER.info(
                f"The results directory {result_directory} does not exist, it will be created"
            )
            os.makedirs(result_directory)

        elif (
                len(os.listdir(result_directory)) > 0
                and not self.language_config["resume_queries"]
        ):
            LOGGER.info(
                f"The result directory {result_directory} already has data in it and the "
                f"configuration has resume queries flag as False, thus either a different"
                f"result directory needs to be provided or the resume queries flag has to be "
                f"set to true if previous training was interrupted"
            )

            raise DuplicateFileError(
                f"The result directory {result_directory} already has data in it and the "
                f"configuration has resume queries flag as False, thus either a different"
                f"result directory needs to be provided or the resume queries flag has to be "
                f"set to true if previous training was interrupted"
            )

        elif (
                self.language_config["resume_queries"]
                and len(os.listdir(result_directory)) > 0
        ):
            LOGGER.info(
                f"The queries will resume, while skipping all the queries present in the folder "
                f"{result_directory}"
            )

    def test_query_folder(self):
        assert os.path.exists(self.query_folder), (
            f"The qwery folder does not exist at the location "
            f"{self.query_folder}, please make sure the query folder is at the "
            f"correct location"
        )

        if len(self.query_folder) < 2:
            LOGGER.warning(
                f"There are only 2 query files in the query path {self.query_folder},"
                f"it is advised to add more queries since loading the ESCO graph is a time consuming "
                f"process and should be done at one go if possible"
            )

    def load_graph(self, turtle_format="n3"):
        assert os.path.exists(
            self.esco_graph_path
        ), f"The turtle file {self.esco_graph_path} does not exist at the given path"

        LOGGER.info(
            f"The file exists at the path, it will now be loaded. This operation might take a while, make "
            f"sure sufficient memory is available for the graph to be loaded"
        )

        self.esco_graph = rdflib.Graph()
        self.esco_graph.parse(self.esco_graph_path, format=turtle_format)

        LOGGER.info(f"The graph is now loaded in memory")

    def run_queries(self):

        LOGGER.info(
            f"The queries will now be loaded, and the resume flag will be checked to see if "
            f"the query fetching should resume or start fresh"
        )

        query_files = sorted(os.listdir(self.query_folder))
        query_limits = self.language_config["limit_data"]

        if query_limits:
            LOGGER.info(
                f"The query limiter is set to true, so only 1000 rows from each queries result will be saved "
                f"in the csv file"
            )

        self.fetch_data(query_files, query_limits)

    def fetch_data(self, query_files, query_limits):

        queries_to_run = query_files
        resume_queries_flag = self.language_config["resume_queries"]
        available_results = []

        if resume_queries_flag:

            LOGGER.info(
                f"The resume flag is set to true, queries for which the results exist will not be run"
            )
            for query_file_name in queries_to_run:
                output_file_name = query_file_name.replace("txt", "tsv")  # todo: this has to fixed in the next update
                if output_file_name in os.listdir(self.output_directory):
                    LOGGER.info(
                        f"The output file {output_file_name} already exists in the output directory "
                        f"{self.output_directory}, and thus will not be recreated again in this case"
                    )
                    available_results.append(query_file_name)

        queries_to_run = list(set(queries_to_run) - set(available_results))

        LOGGER.info(
            f"There are a total of {len(queries_to_run)} queries to run on the graph, they are {queries_to_run}"
        )

        for query_file_name in tqdm(queries_to_run):

            query_file_path = os.path.join(QUERY_FOLDER_PATH, query_file_name)
            with open(query_file_path, "r") as file:
                sparql_query = file.read().rstrip()

            if query_limits:
                sparql_query = sparql_query + f"\nLIMIT 100"

            sparql_query_with_language_tag = sparql_query.replace(
                "LANGUAGE", self.language_used
            )

            self.run_sparql_query(sparql_query_with_language_tag, query_file_name)

    def run_sparql_query(self, sparql_query, query_file_name):

        LOGGER.info(f"The query for {query_file_name} will start now")

        result = self.esco_graph.query(sparql_query)
        LOGGER.info(f"The query for {query_file_name} has finished")

        result_cols = [str("?" + col) for col in list(result.vars)]
        LOGGER.debug(f"The columns for the query are {result_cols}")

        out_file_name = query_file_name.replace("txt", "tsv") # todo: this has to fixed in the next update

        out_file_path = os.path.join(self.output_directory, out_file_name)
        LOGGER.info(f"The path of the outfile is {out_file_path}")

        result_df = pd.DataFrame(result, columns=result_cols)
        LOGGER.debug(f"Results of query are now in the DataFrame")

        result_df.to_csv(out_file_path, sep="\t")
        LOGGER.info(
            f"The results are now written in the output file, the shape of the dataframe written is "
            f"{result_df.shape}"
        )

    def end_function(self):
        LOGGER.info(f'All the queries have now finished, all the data should be available in {self.output_directory}')


data_fetcher = languageDataFetcher()
data_fetcher.check_directories()
data_fetcher.test_query_folder()
data_fetcher.load_graph()
data_fetcher.run_queries()
data_fetcher.end_function()
