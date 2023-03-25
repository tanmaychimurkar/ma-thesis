import logging
import os
import sys
import itertools

import datasets
import numpy as np
import pandas as pd
import json
from datasets import load_metric
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser(description=f'MLM fine tuning description only')

parser.add_argument(
    "--language",
    dest='language',
    required=True,
    help=f'Specify the language for which the fine-tuning is to be done'
)

args = parser.parse_args()

LANGUAGE = args.language

if LANGUAGE == 'en':
    DATASET_DIR = '/cluster/scratch/yakram/sbert-copy/turtle_files/english_data'
    PRETRAINED_MODEL_CHECKPOINT_NAME = 'xlm_roberta-job_ads_checkpoint'
elif LANGUAGE == 'de':
    DATASET_DIR = '/cluster/scratch/yakram/sbert-copy/turtle_files/german_data'
    PRETRAINED_MODEL_CHECKPOINT_NAME = 'xlm_roberta-job_ads_checkpoint'

PRETRAINED_MODEL_CHECKPOINT_PATH = f'/cluster/scratch/yakram/{LANGUAGE}_models/xlm_roberta-job_ads_checkpoint'
LOGFILE_PATH = f'logfiles/mlm_{PRETRAINED_MODEL_CHECKPOINT_NAME}_descriptions_only_{LANGUAGE}.log'
STDOUT_PATH = f'logfiles/mlm_{PRETRAINED_MODEL_CHECKPOINT_NAME}_descriptions_only_{LANGUAGE}.out'
CHECKPOINTING_PATH = f'/cluster/scratch/yakram/{LANGUAGE}_models/{PRETRAINED_MODEL_CHECKPOINT_NAME}_mlm_description_correct'
BEST_CHECKPOINT_PATH = f'/cluster/scratch/yakram/{LANGUAGE}_models/{PRETRAINED_MODEL_CHECKPOINT_NAME}_mlm_description_correct_best'
CHUNK_SIZE = 256

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE_PATH,
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)

f = open(STDOUT_PATH, 'w')
sys.stdout = f

LOGGER.info(f'Starting run, using only concept descriptions for this training')

LOGGER.info(f'Loading models and the tokenizer for the pretrained model {PRETRAINED_MODEL_CHECKPOINT_NAME}')
model = AutoModelForMaskedLM.from_pretrained(PRETRAINED_MODEL_CHECKPOINT_PATH)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_CHECKPOINT_PATH)

LOGGER.info(f'Loading raw datasets')

query_MemConcept_desc = pd.read_csv(os.path.join(
    DATASET_DIR, 'query_MemConcept_description.rq'
), sep='\t')

query_SkillsHier_desc = pd.read_csv(os.path.join(
    DATASET_DIR, 'query_SkillsHier_description.rq'
), sep='\t')

LOGGER.info(f'Getting only the description subsets')

MemConcept_desc = query_MemConcept_desc['?desc'].drop_duplicates()
SkillsHier_desc = query_SkillsHier_desc['?desc'].drop_duplicates()

LOGGER.info(f'Appending desc data and adding temp label column')

desc_data_all = pd.DataFrame(pd.concat([MemConcept_desc, SkillsHier_desc]))
desc_data_all.columns = ['text']
desc_data_all['labels'] = 0

LOGGER.info(f'The combined data has the columns: {desc_data_all.columns}, and shape {desc_data_all.shape}')

LOGGER.info(f'Creating the datasets Dataset object for using in the pipeline')
train_dataset = datasets.Dataset.from_dict(desc_data_all)

LOGGER.info(f'Creating dict from the dataset')
dataset_dict = datasets.DatasetDict({"train": train_dataset})


def tokenize_function(inputs):
    result = tokenizer(inputs['text'])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(idx) for idx in range(len(result["input_ids"]))]
    return result


tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=['text', 'labels'], num_proc=6)

LOGGER.info( f'The total length of the input ids in tokenized dataset is {len(tokenized_datasets["train"]["input_ids"])}')

LOGGER.info(f'The length of a random input in this dataset is {len(tokenized_datasets["train"]["input_ids"][15])}')

LOGGER.info(f'Setting chunk size to concat descriptions longer than 512 tokens')


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // CHUNK_SIZE) * CHUNK_SIZE
    # Split by chunks of max_len
    result = {
        k: [t[i: i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


LOGGER.info(f'Formatting the datasets to keep each input of length 512 tokens')

lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=6)

LOGGER.info(f'After creating chunks, below are the statistics of the data:\nInitial data was in variable '
            f'tokenized_dataset with total tokens being '
            f'{len(list(itertools.chain.from_iterable(tokenized_datasets["train"]["input_ids"])))}, after creating'
            f'chunks, the total tokens in the dataset are '
            f'{len(list(itertools.chain.from_iterable(lm_datasets["train"]["input_ids"])))}')

LOGGER.info(f'Creating train and evaluation datasets for fine-tuning')

total_data = len(lm_datasets['train']['input_ids'])
train_size = int(total_data * 0.9)
test_size = int(total_data - train_size)

lm_datasets_split = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

LOGGER.info(f'Creating the data collator class since there are only descriptions here in this case')
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

metric = load_metric("accuracy")

LOGGER.info(f'Setting training arguments')
training_args = TrainingArguments(output_dir=CHECKPOINTING_PATH,
                                  overwrite_output_dir=True,
                                  do_train=True,
                                  do_eval=True,
                                  num_train_epochs=30,
                                  evaluation_strategy='epoch',
                                  logging_strategy='steps',
                                  logging_steps=1,  # https://github.com/huggingface/transformers/issues/8910,
                                  save_strategy='steps',
                                  save_steps=200,
                                  per_device_train_batch_size=1,
                                  seed=42,
                                  data_seed=42,
                                  eval_steps=1,
                                  load_best_model_at_end=False,
                                  prediction_loss_only=True,
                                  per_gpu_train_batch_size=16
                                  )

LOGGER.info(f'Initiating trainer object')
trainer = Trainer(
    model=model, tokenizer=tokenizer, train_dataset=lm_datasets_split['train'], eval_dataset=lm_datasets_split['test'],
    data_collator=data_collator, args=training_args
)

LOGGER.info(f'Starting training')
trainer_result = trainer.train()

LOGGER.info(f'Saving best model from the best model at end config')
trainer.save_model(BEST_CHECKPOINT_PATH)

### Extra code for loading best model

# from transformers import BertForMaskedLM, BertTokenizer
#
# test_model = BertForMaskedLM.from_pretrained('/mnt/home/tanmaychimurkar/model_saved')
# test_tokenizer = BertTokenizer.from_pretrained('/mnt/home/tanmaychimurkar/model_saved')
#
# # then run test_model.evaluate or predict on the test data to check the accuray, or use the fill mask task from
# # https://huggingface.co/tasks/fill-mask
f.close()