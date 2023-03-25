import itertools
import logging
import sys
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from functools import lru_cache
import ast
from pandarallel import pandarallel

tqdm.pandas()

pandarallel.initialize()

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join('../logfiles/negative_mine_en.log'),
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)

f = open('../logfiles/negative_mine_en.out', 'w')
sys.stdout = f

CURRENT_FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_FILE_LOCATION, '../../turtle_files/english_data/')

LANGUAGE = 'en'

query_MemConcept_narrower = pd.read_csv(os.path.join(DATA_DIR, 'query_MemConcept_Narrower.rq'), sep='\t',
                                        index_col=[0])
query_MemConcept_broader = pd.read_csv(os.path.join(DATA_DIR, 'query_MemConcept_Broader.rq'), sep='\t',
                                       index_col=[0])

SkillsHier_narrower = pd.read_csv(os.path.join(DATA_DIR, 'query_SkillsHier_Narrower.rq'), sep='\t',
                                  index_col=[0])
SkillsHier_broader = pd.read_csv(os.path.join(DATA_DIR, 'query_SkillsHier_Broader.rq'), sep='\t', index_col=[0])

MemConcept_narrower = pd.concat([query_MemConcept_narrower, SkillsHier_narrower])
MemConcept_broader = pd.concat([query_MemConcept_broader, SkillsHier_broader])

skills_only = pd.read_csv(os.path.join(DATA_DIR, 'query_concept_schemes_combined.rq'), sep='\t', index_col=[0])
skills_only = pd.DataFrame(skills_only['?a'])

MemConcept_narrower_skills = skills_only.merge(MemConcept_narrower, on=['?a'], how='left')
MemConcept_broader_skills = skills_only.merge(MemConcept_broader, on=['?a'], how='left')

MemConcept_narrower_skills.dropna(inplace=True)
MemConcept_broader_skills.dropna(inplace=True)

MemConcept_narrower_skills.columns = ['?a', 'linked_uri']
MemConcept_broader_skills.columns = ['?a', 'linked_uri']

MemConcept_narrower_skills_grouped = MemConcept_narrower_skills.groupby(by='?a')['linked_uri'].apply(list).reset_index()
MemConcept_broader_skills_grouped = MemConcept_broader_skills.groupby(by='?a')['linked_uri'].apply(list).reset_index()

merged_data = pd.merge(MemConcept_broader_skills_grouped, MemConcept_narrower_skills_grouped, on=['?a'], how='left')

combined_uris = {}
for idx, value in tqdm(merged_data.iterrows()):
    uri_x = [value['linked_uri_x'] if value['linked_uri_x'] is not np.NAN else None][0]
    uri_y = [value['linked_uri_y'] if value['linked_uri_y'] is not np.NAN else uri_x][0]

    combined_uris[value['?a']] = list(set(uri_x + uri_y))

narrower_broader_lists = list(combined_uris.values())
narrower_broader_combined = list(itertools.chain.from_iterable(narrower_broader_lists))
new_dict = [{k: str(v) for k, v in combined_uris.items()}]

df = pd.DataFrame.from_dict(new_dict[0], orient='index').reset_index()
df.columns = ['main_uri', 'broader_narrower_uri']


# df['broader_narrower_uri'] = df['broader_narrower_uri'].apply(lambda x: ast.literal_eval(x))

@lru_cache
def mine_negatives(row):
    print(row)
    row = ast.literal_eval(row)

    negative_uris = []
    remaining_uris = list(set(narrower_broader_combined) - set(row))

    for uri, value in tqdm(combined_uris.items()):

        if set(value).issubset(remaining_uris):
            negative_uris.append(uri)

    return negative_uris


df['negative_uris'] = df['broader_narrower_uri'].parallel_apply(mine_negatives)

#
df.to_csv(os.path.join(DATA_DIR, '../', 'negative_mined_script_last.csv'),
          sep='\t')

#
#
#
# def mine_negatives(x):
#     negative_uris = []
#     remaining_uris = list(set(narrower_broader_unique_list) - set(x))
#
#     for record in tqdm(narrower_broader_mapping):
#         main_uri_narrower_broader = record['linked_uri']
#
#         if set(main_uri_narrower_broader).issubset(remaining_uris):
#             negative_uris.append(record['?a'])
#
#     return negative_uris
#
#
# narrower_broader_combined_grouped['negative_uris'] = narrower_broader_combined_grouped['linked_uri'].parallel_apply(
#     mine_negatives)
#
# narrower_broader_combined_grouped.to_csv(os.path.join(DATA_DIR, '../', 'narrower_broader_combined_grouped_renewed.csv'),
#                                          sep='\t')
# LOGGER.info(f'Negatives are mined and stored')
f.close()
