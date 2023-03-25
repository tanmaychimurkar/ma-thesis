
#german BERT model without domain adaption is 'deepset/gbert-base', with domain adaption to our job ads 'agne/jobGBERT'

#tsdae
python -u training_tsdae_from_file.py -m 'agne/jobGBERT' -s 'data/adspans-ontospans.txt' -p 'cls' -d 0.4 ;

#sts
python -u training_stsbenchmark.py -m 'models/tsdae_agne-jobGBERT_adspans-ontospans_cls_dR04' -b 16 -e 1 -p mean ;

#mnr
python -u training_mNR.py -m models/sts-mean-bS16_tsdae_agne-jobGBERT_adspans-ontospans_cls_dR04  -s data/anchorsPositivesPerUri.txt -b 32 -e 1 -p cls ; 

