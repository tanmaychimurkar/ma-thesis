import argparse
import logging
import os
import sys

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch.optim
# from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import losses
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.readers import InputExample
from tqdm import tqdm
from evaluation.custom_triplet_evaluator import TripletEvaluatorTensorboard
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/cluster/scratch/yakram/conda_env/'

# accelerator = Accelerator()

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser(description=f'MLM fine tuning description only')

parser.add_argument(
    "--language",
    dest='language',
    required=True,
    help=f'Specify the language for which the fine-tuning is to be done'
)

parser.add_argument(
    "--dataset_name",
    dest='dataset_name',
    required=True,
    help=f'Specify the dataset from which the fine-tuning is to be done'
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
    "--batch_size",
    dest='batch_size',
    required=True,
    help=f'Specify the batch size for training'
)

parser.add_argument(
    "--pretrained_mode",
    dest='pretrained_mode',
    required=True,
    help=f'Specify if a pretrained mode is to be used out of the box or to use our fine-tuned models'
)

parser.add_argument(
    "--pretrained_model",
    dest='pretrained_model',
    required=True,
    help=f'Specify if a pretrained model is to be used out of the box or to use our fine-tuned models'
)

parser.add_argument(
    "--outdir",
    dest='outdir',
    required=True,
    help=f'Specify the output directory where the model is to be stored'
)
args = parser.parse_args()

LANGUAGE = args.language

DATASET_DIR = f'/cluster/scratch/yakram/sbert-copy/turtle_files/mnr_data/{LANGUAGE}'
MODEL_BASE_DIR = f'/cluster/scratch/yakram/{LANGUAGE}_models/'

LOGFILE_PATH = f'logfiles/{LANGUAGE}_{args.outdir}.log'
STDOUT_PATH = f'logfiles/{LANGUAGE}_{args.outdir}.out'

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename=LOGFILE_PATH,
    filemode="w",
    format="%(levelname)s:[%(filename)s:%(lineno)d] - %(message)s [%(asctime)s]",
)

f = open(STDOUT_PATH, 'w')
sys.stdout = f

DATASET_PATH = os.path.join(DATASET_DIR, args.dataset_name + '.csv')
DEV_DATASET = os.path.join(DATASET_DIR, 'hard_neg_dev.csv')


if args.pretrained_mode != 'True':
    MODEL_CHECKPOINT_PATH = os.path.join(MODEL_BASE_DIR, args.model_dir, args.model_checkpoint)
    model_checkpoint_name = args.model_checkpoint
else:
    MODEL_CHECKPOINT_PATH = args.pretrained_model
    model_checkpoint_name = MODEL_CHECKPOINT_PATH

CHECKPOINT_PATH = os.path.join(MODEL_BASE_DIR, args.outdir + '/checkpoint')
OUTPUT_PATH = os.path.join(MODEL_BASE_DIR, args.outdir)

LOGGER.info(f'The following variables have been loaded\nModel Checkpoint Path:{MODEL_CHECKPOINT_PATH}\n'
            f'Dataset Path:{DATASET_PATH}\n')

# device = accelerator.device

LOGGER.info(f'The model is loaded')
word_embedding_model = models.Transformer(MODEL_CHECKPOINT_PATH, cache_dir='/cluster/scratch/yakram/conda_env/')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'CLS')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
LOGGER.info(f'The model is loaded')
# model.to(device)

epochs = int(args.epochs)
batch_size = int(args.batch_size)

LOGGER.info(f'The epochs for the model are {epochs}, and the batch size is {batch_size}')

dataset_df = pd.read_csv(DATASET_PATH, sep='\t', index_col=[0])
dev_data = pd.read_csv(DEV_DATASET, sep='\t', index_col=[0])

if LANGUAGE == 'en':
    dev_data = dev_data.sample(n=1500)

if len(dataset_df.columns) > 3 and '?a' in dataset_df.columns:
    dataset_df.drop(columns=['?a'], inplace=True)

if len(dev_data.columns) > 3 and '?a' in dev_data.columns:
    dev_data.drop(columns=['?a'], inplace=True)

dataset_df.dropna(inplace=True)
dataset_df.reset_index(drop=True, inplace=True)

dev_data.dropna(inplace=True)
dev_data.reset_index(drop=True, inplace=True)

dataset_df_columns = dataset_df.columns
train_text_pairs = []
if len(dataset_df_columns) == 2:
    for idx, value in tqdm(dataset_df.iterrows()):
        input_example = InputExample(texts=[value[dataset_df_columns[0]], value[dataset_df_columns[1]]])
        train_text_pairs.append(input_example)
elif len(dataset_df_columns) == 3:
    for idx, value in tqdm(dataset_df.iterrows()):
        input_example = InputExample(texts=[value[dataset_df_columns[0]], value[dataset_df_columns[1]],
                                            value[dataset_df_columns[2]]])
        train_text_pairs.append(input_example)

dev_df_columns = dev_data.columns
dev_text_pairs = []
if len(dev_df_columns) == 2:
    for idx, value in tqdm(dev_data.iterrows()):
        input_example = InputExample(texts=[value[dev_df_columns[0]], value[dev_df_columns[1]]])
        dev_text_pairs.append(input_example)
elif len(dev_df_columns) == 3:
    for idx, value in tqdm(dev_data.iterrows()):
        input_example = InputExample(texts=[value[dev_df_columns[0]], value[dev_df_columns[2]],
                                            value[dev_df_columns[1]]])
        dev_text_pairs.append(input_example)

if len(train_text_pairs) < 10:
    raise ValueError(f'There is not enough training data, either the data file is corrupted or there is something '
                     f'wrong with the logic')

LOGGER.info(
    f'The text pairs have been loaded, there are a total of {len(train_text_pairs)} text pairs for model fine tuning')

train_dataloader = NoDuplicatesDataLoader(train_examples=train_text_pairs, batch_size=batch_size)
LOGGER.info(f'The train dataloader has been created')
training_loss = losses.MultipleNegativesRankingLoss(model)

dev_evaluator = TripletEvaluatorTensorboard.from_input_examples(dev_text_pairs, main_distance_function=0,
                                                                write_csv=True,
                                                                name=f'mnr-{args.dataset_name}-{model_checkpoint_name}-'
                                                                     f'{args.batch_size}',
                                                                show_progress_bar=True,
                                                                tb_logdir=f'{OUTPUT_PATH}/runs',
                                                                tb_comment=f'mnr-{args.dataset_name}-'
                                                                           f'{model_checkpoint_name}-{args.batch_size}')

# model, optimizer, train_dataloader = accelerator.prepare(
#     model, torch.optim.adamw, train_dataloader
# )

model.fit(train_objectives=[(train_dataloader, training_loss)],
          epochs=epochs,
          warmup_steps=2,
          checkpoint_path=CHECKPOINT_PATH,
          evaluator=dev_evaluator,
          evaluation_steps=1,
          checkpoint_save_steps=200,
          output_path=OUTPUT_PATH
          )

evaluation_file = f'triplet_evaluation_mnr-{args.dataset_name}-{model_checkpoint_name}-{args.batch_size}_results.csv'

evaluation_file_path = os.path.join(OUTPUT_PATH, 'eval', evaluation_file)

summary_df = pd.read_csv(evaluation_file_path)

max_step = summary_df.steps.max()
summary_df['steps'] = summary_df['steps'].apply(lambda x: max_step + 1 if x == -1 else x)
summary_df_subset = summary_df[list(summary_df.columns)[1:]]
summary_df_subset_grouped = summary_df_subset.groupby(by='steps').mean().reset_index()

writer = SummaryWriter(log_dir=f'{MODEL_BASE_DIR}mnr_tensorboard_runs/{args.outdir}')
for idx, value in summary_df_subset_grouped.iterrows():
    writer.add_scalar('Accuracy/accuracy_cosinus',
                      value['accuracy_cosinus'],
                      value['steps'])

    writer.add_scalar('Accuracy/accuracy_manhattan',
                      value['accuracy_manhattan'],
                      value['steps'])

    writer.add_scalar('Accuracy/accuracy_euclidean',
                      value['accuracy_euclidean'],
                      value['steps'])

    writer.add_scalar('Accuracy/epochs',
                      0,
                      value['steps'])

writer.close()

LOGGER.info(f'Saved tensorboard run logs in {MODEL_BASE_DIR}/mnr_runs/{args.outdir}')

LOGGER.info(f'Model training is complete')
