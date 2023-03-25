"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

TSDAE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_tsdae_from_file.py path/to/sentences.txt

"""
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, datasets, losses
import logging
import gzip
from torch.utils.data import DataLoader
import tqdm
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])



parser = argparse.ArgumentParser()
parser.add_argument("-m", "-modelname", dest="modelname", help="modelfile")
parser.add_argument("-s", "-senfile", dest="sentencefile", help="sentencefile")
parser.add_argument("-p", "-pooling", dest="pooling", help="pooling method")
parser.add_argument("-d", "-delrate", dest="delrate", help="deleation rate")
args = parser.parse_args()



model_output_path  = 'models/tsdae_' +  args.modelname.replace('/', '-') + '_' + args.sentencefile.replace('data/', '').replace('.txt', '').replace('_', '-') + '_' + args.pooling + '_dR' + args.delrate.replace('.', '')
print(model_output_path)

################# Read the train corpus  #################
train_sentences = []
filepath = args.sentencefile
with gzip.open(args.filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, desc='Read file'):
        line = line.strip()
        if len(line) >= 10 and len(line) <=300:
            train_sentences.append(line)


logging.info("{} train sentences".format(len(train_sentences)))

################# Intialize an SBERT model #################

word_embedding_model = models.Transformer(args.modelname)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


################# Train and evaluate the model (it needs about 1 hour for one epoch of AskUbuntu) #################
# We wrap our training sentences in the DenoisingAutoEncoderDataset to add deletion noise on the fly
batch_size = 8
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=args.modelname, tie_encoder_decoder=True)


logging.info("Start training")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    #steps_per_epoch=10,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True,
    checkpoint_path=model_output_path,
    checkpoint_save_steps = 10000,
    #checkpoint_save_total_limit=3,
    output_path=model_output_path,
    use_amp=False               #Set to True, if your GPU supports FP16 cores
)
