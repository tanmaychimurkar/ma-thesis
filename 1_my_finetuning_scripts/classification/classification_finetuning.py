import argparse
import logging
import os
import sys

import datasets
import numpy as np
import pandas as pd
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_CACHE'] = '/cluster/scratch/yakram/conda_env/'
os.environ['HF_HOME'] = '/cluster/scratch/yakram/conda_env/'

parser = argparse.ArgumentParser(description=f'Classification fine tuning term-term and term-desc, randomly shuffled')

parser.add_argument(
    "--language",
    dest='language',
    required=True,
    help=f'Specify the language for which the fine-tuning is to be done'
)

parser.add_argument(
    "--model_dir",
    dest='model_dir',
    required=True,
    help=f'Specify the dir of the model in which the checkpoint is preset'
)

parser.add_argument(
    "--model_checkpoint",
    dest='model_checkpoint',
    required=True,
    help=f'Specify the checkpoint name of the model in which the checkpoint is preset'
)

parser.add_argument(
    "--epochs",
    dest='epochs',
    required=True,
    help=f'Specify the number of epochs to run the training for'
)

parser.add_argument(
    "--pretrained_mode",
    dest='pretrained_mode',
    required=True,
    help=f'Specify if a pretrained model is to be used'
)

parser.add_argument(
    "--batch_size",
    dest='batch_size',
    required=True,
    help=f'Specify the batch size for finetuning'
)

parser.add_argument(
    "--outdir",
    dest='outdir',
    required=True,
    help=f'Specify the output directory where the model is to be stored'
)


args = parser.parse_args()

LANGUAGE = args.language
DATASET_DIR = f'/cluster/scratch/yakram/sbert-copy/turtle_files/classification_data/{LANGUAGE}'
MODEL_BASE_DIR = f'/cluster/scratch/yakram/{LANGUAGE}_models/'

LOGFILE_PATH = f'logfiles/{LANGUAGE}_{args.outdir}.log'
STDOUT_PATH = f'logfiles/{LANGUAGE}_{args.outdir}.out'

f = open(STDOUT_PATH, 'w')
sys.stdout = f

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE_PATH,
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)
LOGGER.info(f'Starting run, using term-term and term-desc data for classification')


CHECKPOINT_PATH = os.path.join(MODEL_BASE_DIR, args.model_dir, args.model_checkpoint)
OUTPUT_PATH = os.path.join(MODEL_BASE_DIR, args.outdir)
CLASSIFICATION_FILES = 'shuffled_classification_data.csv'

if args.pretrained_mode == 'true':
    CHECKPOINT_PATH = 'xlm-roberta-base' # todo: this has to be changed later

LOGGER.info(f'The following directories are used:\nCHECKPOINT_PATH: {CHECKPOINT_PATH},\nOUTPUT_PATH: {OUTPUT_PATH},\n'
            f'CLASSIFICATION_FILE: {CLASSIFICATION_FILES}')

classification_data = pd.read_csv(os.path.join(DATASET_DIR, CLASSIFICATION_FILES), sep='\t', index_col=[0])
classification_data.reset_index(drop=True, inplace=True)
classification_data.drop(columns=['?a'], inplace=True)
classification_data.rename(columns={'labels': 'label'}, inplace=True)
labels = list(classification_data['label'].unique())

LOGGER.info(f'The classification data is loaded, there are a shape of data is{classification_data.shape} '
            f'and the number of labels are {len(labels)}')

LOGGER.info(f'Loading the tokenizer and the model checkpoint from the location: {CHECKPOINT_PATH}')

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, cache_dir='/cluster/scratch/yakram/conda_env/')
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH, num_labels=len(labels), cache_dir='/cluster/scratch/yakram/conda_env/')

train_dataset = datasets.Dataset.from_pandas(classification_data)
dataset_dict = datasets.DatasetDict({"train": train_dataset})

LOGGER.info(f'The train dataset dict has been created')

sentence1_key, sentence2_key = ('term1', 'term2')


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,
                     padding=True)


metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


encoded_dataset = dataset_dict.map(preprocess_function, batched=True, num_proc=4)
LOGGER.info(f'The dataset has been encoded')

total_data = len(encoded_dataset['train']['input_ids'])
train_size = int(total_data * 0.9)
test_size = int(total_data - train_size)

dataset_splits = encoded_dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

LOGGER.info(f'Created the train test splits from the dataset')

args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    num_train_epochs=int(args.epochs),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy='steps',
    # save_steps='epoch',
    # eval_steps=150,
    logging_steps=1,
    seed=42,
    data_seed=42,
    per_device_train_batch_size=int(args.batch_size),
    per_device_eval_batch_size=int(args.batch_size),
)

trainer = Trainer(
    model,
    args,
    train_dataset=dataset_splits["train"],
    eval_dataset=dataset_splits['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

LOGGER.info(f'The training has finished')
