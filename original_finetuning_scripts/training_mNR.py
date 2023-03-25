"""
This scripts demonstrates how to train a sentence embedding model for Information Retrieval.

As dataset, we use Quora Duplicates Questions, where we have pairs of duplicate questions.

As loss function, we use MultipleNegativesRankingLoss. Here, we only need positive pairs, i.e., pairs of sentences/texts that are considered to be relevant. Our dataset looks like this (a_1, b_1), (a_2, b_2), ... with a_i / b_i a text and (a_i, b_i) are relevant (e.g. are duplicates).

MultipleNegativesRankingLoss takes a random subset of these, for example (a_1, b_1), ..., (a_n, b_n). a_i and b_i are considered to be relevant and should be close in vector space. All other b_j (for i != j) are negative examples and the distance between a_i and b_j should be maximized. Note: MultipleNegativesRankingLoss only works if a random b_j is likely not to be relevant for a_i. This is the case for our duplicate questions dataset: If a sample randomly b_j, it is unlikely to be a duplicate of a_i.


The model we get works well for duplicate questions mining and for duplicate questions information retrieval. For question pair classification, other losses (like OnlineConstrativeLoss) work better.
"""

from torch.utils.data import DataLoader
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, models
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
from zipfile import ZipFile
import random
import argparse
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout



parser = argparse.ArgumentParser()
parser.add_argument("-m", "-modelname", dest="modelname", help="modelfile")
parser.add_argument("-s", "-senfile", dest="sentencefile", help="sentencefile")
parser.add_argument("-b", "-batchsize", dest="batchsize", help="batchsize")
parser.add_argument("-e", "-epochs", dest="epochs", help="epochs")
parser.add_argument("-p", "-pooling", dest="pooling", help="pooling")

args = parser.parse_args()



model_output_path  = 'models/mnr-' + args.pooling + '-bS'  + args.batchsize  + '_'+  args.modelname.replace('models/', '-') + '_' + args.sentencefile.replace('data/', '').replace('.txt', '').replace('_', '-')
print(model_output_path)

#new:
word_embedding_model = models.Transformer(args.modelname)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])




#Training for multiple epochs can be beneficial, as in each epoch a mini-batch is sampled differently
#hence, we get different negatives for each positive
num_epochs = int(args.epochs)

#Increasing the batch size improves the performance for MultipleNegativesRankingLoss. Choose it as large as possible
#I achieved the good results with a batch size of 300-350 (requires about 30 GB of GPU memory)
train_batch_size = args.batchsize

dataset_path = args.sentencefile
model_save_path = model_output_path

os.makedirs(model_save_path, exist_ok=True)


######### Read train data  ##########
# train_samples = []
# with open(dataset_path, encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#     #    if row['is_duplicate'] == '1':
#             train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=1))
#             train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=1)) #if A is a duplicate of B, then B is a duplicate of A

TEST=False
train_samples = []
counter = 0
with open(dataset_path, 'rt', encoding='utf-8') as f:
    counter +=1
    for line in f:
        counter +=1
        if counter % 2500 == 0 and TEST == True:
            break
        l = line.strip().split('\t')

        i = InputExample(texts=[l[0], l[1]])
        train_samples.append(i)

#print(train_samples)
# After reading the train_samples, we create a DataLoader

batch_size = int(args.batchsize)

#train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

train_dataloader = NoDuplicatesDataLoader(train_samples, batch_size=batch_size)
print(len(train_dataloader))
train_loss = losses.MultipleNegativesRankingLoss(model)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=10,
          checkpoint_path=model_output_path,
          checkpoint_save_steps = 1000,
          output_path=model_save_path
          )
