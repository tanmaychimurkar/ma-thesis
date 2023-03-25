import logging
import sys
import pandas as pd
from tqdm import tqdm
import os

tqdm.pandas()
from pandarallel import pandarallel

pandarallel.initialize()

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join('../logfiles/negative_mine_de.log'),
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)

f = open('../logfiles/negative_mine_de.out', 'w')
sys.stdout = f

CURRENT_FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(CURRENT_FILE_LOCATION, '../../turtle_files/german_data/')

LANGUAGE = 'de'

query_SkillsHier_Broader = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_Broader.rq', index_col='Unnamed: 0',
                                       delimiter='\t')
query_SkillsHier_altLabel = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_altLabel.rq', index_col='Unnamed: 0',
                                        delimiter='\t')
query_MemConcept_hiddenLabel = pd.read_csv(RESULTS_PATH + 'query_MemConcept_hiddenLabel.rq', index_col='Unnamed: 0',
                                           delimiter='\t')
query_SkillsHier_Schemes = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_Schemes.rq', index_col='Unnamed: 0',
                                       delimiter='\t')
query_MemConcept_BroaderTransitive = pd.read_csv(RESULTS_PATH + 'query_MemConcept_BroaderTransitive.rq',
                                                 index_col='Unnamed: 0', delimiter='\t')
query_MemConcept_description = pd.read_csv(RESULTS_PATH + 'query_MemConcept_description.rq', index_col='Unnamed: 0',
                                           delimiter='\t')
query_SkillsHier_description = pd.read_csv(RESULTS_PATH + 'query_MemConcept_description.rq', index_col='Unnamed: 0',
                                           delimiter='\t')  # todo: change the name of thef ifle here
query_MemConcept_TopConceptOf = pd.read_csv(RESULTS_PATH + 'query_MemConcept_TopConceptOf.rq', index_col='Unnamed: 0',
                                            delimiter='\t')
query_SkillsHier_Narrower = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_Narrower.rq', index_col='Unnamed: 0',
                                        delimiter='\t')
query_MemConcept_Broader = pd.read_csv(RESULTS_PATH + 'query_MemConcept_Broader.rq', index_col='Unnamed: 0',
                                       delimiter='\t')
query_MemConcept_isOptionalSkillFor = pd.read_csv(RESULTS_PATH + 'query_MemConcept_isOptionalSkillFor.rq',
                                                  index_col='Unnamed: 0', delimiter='\t')
query_MemConcept_altLabel = pd.read_csv(RESULTS_PATH + 'query_MemConcept_altLabel.rq', index_col='Unnamed: 0',
                                        delimiter='\t')
query_MemConcept_RelatedOptionalSkill = pd.read_csv(RESULTS_PATH + 'query_MemConcept_RelatedOptionalSkill.rq',
                                                    index_col='Unnamed: 0', delimiter='\t')
query_SkillsHier_prefLabel = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_prefLabel.rq', index_col='Unnamed: 0',
                                         delimiter='\t')
query_MemConcept_prefLabel = pd.read_csv(RESULTS_PATH + 'query_MemConcept_prefLabel.rq', index_col='Unnamed: 0',
                                         delimiter='\t')
query_SkillsHier_BroaderTransitive = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_BroaderTransitive.rq',
                                                 index_col='Unnamed: 0', delimiter='\t')
query_MemConcept_isEssentialSkillFor = pd.read_csv(RESULTS_PATH + 'query_MemConcept_isEssentialSkillFor.rq',
                                                   index_col='Unnamed: 0', delimiter='\t')
query_MemConcept_skillReuseLevel = pd.read_csv(RESULTS_PATH + 'query_MemConcept_skillReuseLevel.rq',
                                               index_col='Unnamed: 0', delimiter='\t')
query_MemConcept_Narrower = pd.read_csv(RESULTS_PATH + 'query_MemConcept_Narrower.rq', index_col='Unnamed: 0',
                                        delimiter='\t')
query_SkillsHier_hiddenLabel = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_hiddenLabel.rq', index_col='Unnamed: 0',
                                           delimiter='\t')
query_MemConcept_Schemes = pd.read_csv(RESULTS_PATH + 'query_MemConcept_Schemes.rq', index_col='Unnamed: 0',
                                       delimiter='\t')
query_MemConcept_SkillType = pd.read_csv(RESULTS_PATH + 'query_MemConcept_skillType.rq', index_col='Unnamed: 0',
                                         delimiter='\t')
query_SkillsHier_SkillType = pd.read_csv(RESULTS_PATH + 'query_SkillsHier_skillType.rq', index_col='Unnamed: 0',
                                         delimiter='\t')

skills_only = pd.read_csv(os.path.join(RESULTS_PATH, 'query_concept_schemes_combined.rq'), sep='\t', index_col=[0])
skills_only = pd.DataFrame(skills_only['?a'])

query_MemConcept_Narrower = skills_only.merge(query_MemConcept_Narrower, on=['?a'], how='left')
query_MemConcept_Broader = skills_only.merge(query_MemConcept_Broader, on=['?a'], how='left')

Narrower_groupby = query_MemConcept_Narrower.groupby(by='?a')['?narroweruri'].apply(list).reset_index()
Broader_groupby = query_MemConcept_Broader.groupby(by='?a')['?broaderuri'].apply(list).reset_index()

Narrower_Broader_combined = Narrower_groupby.merge(Broader_groupby, on='?a', how='left')
Narrower_Broader_combined['all_uri_combined'] = Narrower_Broader_combined['?narroweruri'] + Narrower_Broader_combined[
    '?broaderuri']

narroweruri_unique = query_MemConcept_Narrower['?narroweruri'].unique().tolist()
broaderuri_unique = query_MemConcept_Broader['?broaderuri'].unique().tolist()
narrower_broader_uris_combined = list(set(narroweruri_unique + broaderuri_unique))
narrower_broader_mapping = Narrower_Broader_combined[['?a', 'all_uri_combined']].to_dict(orient='records')


def mine_negatives(x):
    negative_uris = []
    remaining_uris = list(set(narrower_broader_uris_combined) - set(x))

    for record in narrower_broader_mapping:
        main_uri_narrower_broader = record['all_uri_combined']

        if set(main_uri_narrower_broader).issubset(remaining_uris):
            negative_uris.append(record['?a'])

    return negative_uris


Narrower_Broader_combined['negative_uris'] = Narrower_Broader_combined['all_uri_combined'].parallel_apply(
    mine_negatives, progress_bar=True)

Narrower_Broader_combined.to_csv(os.path.join(RESULTS_PATH, '../', 'negative_mined_de.csv'), sep='\t')
LOGGER.info(f'Negatives are mined and stored')
f.close()
