"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from datasets import load_dataset
import argparse
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



parser = argparse.ArgumentParser()
parser.add_argument("-m", "-modelname", dest="modelname", help="modelfile")
parser.add_argument("-b", "-batchsize", dest="batchsize", help="batchsize")
parser.add_argument("-e", "-epochs", dest="epochs", help="epochs")
parser.add_argument("-p", "-pooling", dest="pooling", help="pooling")

args = parser.parse_args()
#Check if dataset exsist. If not, download and extract  it
#sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

#if not os.path.exists(sts_dataset_path):
#    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)



#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = args.modelname #if len(sys.argv) > 1 #else 'distilbert-base-uncased'
model_name_out = 'models/sts-' + args.pooling + '-bS' + args.batchsize + '_' + model_name.replace('models/', '').replace('/', '-')
print(model_name)
print(model_name_out)

#train_tsdae-bert-base-german-cased-spans-2022-07-25_14-45-02
# Read the dataset
train_batch_size = int(args.batchsize) #16
num_epochs = int(args.epochs )#4
#model_save_path = 'output/training_stsbenchmark_'+model_name_out +'_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#model_save_path = os.path.join('output', model_name_out)

#print(model_save_path)
#Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)


#Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                              pooling_mode_mean_tokens=True,
                              pooling_mode_cls_token=False,
                              pooling_mode_max_tokens=False)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []

#dataset = load_dataset("stsb_multi_mt", name="de")
dataset_train = load_dataset("stsb_multi_mt", name="de", split='train')
dataset_dev = load_dataset("stsb_multi_mt", name="de", split='dev')
dataset_test = load_dataset("stsb_multi_mt", name="de", split='test')

print(dataset_train)
print(dataset_train.features)
print('test', len(dataset_test))
print('dev', len(dataset_dev))

print('train', len(dataset_train))

#print(type(dataset))
#print(dataset)
#print(dataset.features)
#with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    #for row in reader:

for row in dataset_train:
    score = float(row['similarity_score']) / 5.0  # Normalize score to range 0 ... 1
    #if score < 0.5:
    #    print(score)
    inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
    train_samples.append(inp_example)

for row in dataset_dev:
    score = float(row['similarity_score']) / 5.0  # Normalize score to range 0 ... 1
    inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
    dev_samples.append(inp_example)


for row in dataset_test:
    score = float(row['similarity_score']) / 5.0  # Normalize score to range 0 ... 1
    inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
    test_samples.append(inp_example)




# for row in dataset:
#     print(row)
#     try:
#         score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
#         inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
#
#         if row['split'] == 'dev':
#             dev_samples.append(inp_example)
#         elif row['split'] == 'test':
#             test_samples.append(inp_example)
#         else:
#             train_samples.append(inp_example)
#     except:
#         print('row not processed:', row)


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          checkpoint_save_steps = 180,
          checkpoint_path=model_name_out,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_name_out)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

#model = SentenceTransformer(model_save_path)
#test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
#test_evaluator(model)#, output_path=model_save_path)
