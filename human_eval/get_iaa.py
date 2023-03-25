

import pandas as pd
import numpy as np
import krippendorff as kd




edu = ['** Biologielaborant/in **',
'** BWL-Studium **', '** MBA **', '** MTRA **', '** Handwerkliche **', '** Feuerwehr ** Sanität',
'** Disponent **', 'Wirtschaftsinformatik, Informatik, Betriebswirtschaft, IT ** Mathematik **',  '** Konstrukteur **',
'Financial Consulting ** Bank ** Management']


exp = ['Wirtschaftsprüfung, Big4, Banking, Private Banking ** Asset- ** Fond Management',
'Internet Sicherheit Technologien ** VPN ** IP Security',
'funktionellen Behandlung ** Neurotraining **',
'.NET, OO-Programmierung, DB-Lösungen ** MS-SQL ** 3-Tier-Architektur, Internet-Programmierung',
'** Lebensmittelverkauf **',
'** Gewinnung von Neukunden ** Akquise',
'** IT Spektrum **',
'** Führungserfahrung **',
'** öffentliche Verwaltung **',
 '** Bauingenieurwesen **']


#wreite separate functions for class, uri, term, that can red in dict and give only new items to annoated.

lng = ['** Sehr gute Kenntnisse in Deutsch und Französisch **',
	'** Muttersprache Deutsch **', '** Fremdsprachenkenntnissen **', '** Französisch **', '** Englisch (sehr gut in Wort und Schrift) **']


for level in ['class', 'uri']:

    print('----------------------------')
    print('level:', level)

    if level == 'class':
        for sample in ['challenge', 'random']:
            print('sample:', sample)
            if sample == 'random':

                df = pd.read_csv('/Users/ann_so/GitHub/sbert_adaption/human_eval/random_sample_classes_evaluated_asg_eb_cs_mean.csv')
                df.jobadterm = df.jobadterm.str.strip()
                fullist = edu + exp + lng

                for i in fullist:
                    i = [i]
                    df_sel =df[df['jobadterm'].isin(i)]
                    lc = [c for c in df_sel.columns.values if c.startswith('annotator')]
                    labels = df_sel[lc]
                    lT = labels.T
                    try:
                        r = kd.alpha(lT, level_of_measurement='ordinal')
                        print(i, r)
                    except:
                        print('except:', i,  1)

                for i, l in zip(['edu', 'exp', 'lng', 'all'], [edu, exp, lng, fullist]):
                    df_sel =df[df['jobadterm'].isin(l)]
                    lc = [c for c in df_sel.columns.values if c.startswith('annotator')]
                    labels = df_sel[lc]
                    lT = labels.T
                    r = kd.alpha(lT, level_of_measurement='ordinal')
                    print(i, r)

            elif sample == 'challenge':
                df = pd.read_csv('challenge_sample_classes_evaluated_asg_eb_cs_mean.txt', sep='\t')
                df.jobadterm = df.jobadterm.str.strip()
                fullist = list(df['jobadterm'].unique())

                print(fullist)
                for i in fullist:
                    i = [i]
                    df_sel =df[df['jobadterm'].isin(i)]
                    lc = [c for c in df_sel.columns.values if c.startswith('annotator')]
                    labels = df_sel[lc]
                    lT = labels.T

                    try:
                        r = kd.alpha(lT, level_of_measurement='ordinal')
                        print(i, r)
                    except:
                        print('except:', i,  1)
                for i, l in zip(['all'], [fullist]):
                    df_sel =df[df['jobadterm'].isin(l)]
                    lc = [c for c in df_sel.columns.values if c.startswith('annotator')]
                    labels = df_sel[lc]
                    lT = labels.T
                    r = kd.alpha(lT, level_of_measurement='ordinal')
                    print(i, r)


    elif level == 'uri':
        print('sample: random' )
        #for evaluation in ['regular', 'soft', 'hard']:
        #    print('evaluation mode', evaluation)

        df = pd.read_csv('/Users/ann_so/GitHub/sbert_adaption/human_eval/random_sample_uris_evaluated_asg_eb_mean.csv', sep=',')
        # if evaluation  == 'soft':
        #     df['annotator_x'].replace({0.5: 1.0}, inplace=True)
        #     df['annotator_y'].replace({0.5: 1.0}, inplace=True)
        # elif evaluation  == 'hard':
        #     df['annotator_x'].replace({0.5: 0.0}, inplace=True)
        #     df['annotator_y'].replace({0.5: 0.0}, inplace=True)
        # elif evaluation == 'regular':
        #     pass
        fullist = edu +  exp+ lng

        for i, l in zip(['edu', 'exp', 'lng', 'all'], [edu, exp, lng, fullist]):
            df_sel =df[df['jobadterm'].isin(l)]
            lc = [c for c in df_sel.columns.values if c.startswith('annotator')]
            labels = df_sel[lc]
            lT = labels.T
            r = kd.alpha(lT, level_of_measurement='ordinal')
            print(i, r)

        for i in fullist:
            i = [i]
            df_sel =df[df['jobadterm'].isin(i)]
            lc = [c for c in df_sel.columns.values if c.startswith('annotator')]
            labels = df_sel[lc]
            lT = labels.T
            try:
                r = kd.alpha(lT, level_of_measurement='ordinal')
                print(i, r)
            except:
                print('except:', i,  1)
