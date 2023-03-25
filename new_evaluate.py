import json
from smart_open import open
import os
import numpy as np
import pandas as pd


def get_fullterm(d):
    "convert term to fullterm for reading in evaldicts etc."
    if 'termJobad' in d:
        term = d['termJobad']
    else:

        term = d['term'].replace('\n', ' ')
    prev = d['termsPrevious'].replace('\n', ' ')
    next = d['termsNext'].replace('\n', ' ')
    fullterm = prev + ' ** ' + term + ' ** ' + next
    fullterm = fullterm.strip()
    return fullterm


def get_classlabels(infile):
    "get original labels for numeric values"
    with open(infile, 'r') as f:
        labels = {}
        for line in f:
            d = json.loads(line)
            # labels[d['label_numeric']] = d['label_original']
            labels[d['label_numeric']] = d['label_replaced']

    return labels


def get_uripreflabels(infile):
    "get for each uri a pref label (deMixed or deMale for occupation terms)"
    uri2pref = {}
    with open(infile, 'r') as f:
        for line in f:
            d = json.loads(line)
            u = d['uri']
            if u in uri2pref:
                continue
            if 'termType' in d and d['termType'] == 'prefLabel' and 'termLangGen' not in d:
                uri2pref[u] = d['term']
            elif 'termType' in d and 'termLangGen' in d and d['termLangGen'] in ['deMixed', 'deMale']:
                uri2pref[u] = d['term']
    return uri2pref


def get_evaldict_terms(evalfiles):
    "read in annotations. only one annoation per file, from last column. (if several annoataros, aggregate before)"
    t2e = {}
    for evalfile in evalfiles:
        with open(evalfile, 'r') as f:
            for line in f:
                if line.startswith('jobadterm'):
                    continue
                l = line.rstrip().split('\t')
                term = l[0].strip()
                #	print('.-------')
                #	print(l)
                #	print(t)
                if term in t2e:
                    t2e[term][l[1]] = float(l[-1])
                else:
                    t2e[term] = {l[1]: float(l[-1])}
    return t2e


edu = ['** Biologielaborant/in **',
       '** BWL-Studium **', '** MBA **', '** MTRA **', '** Handwerkliche **', '** Feuerwehr ** Sanität',
       '** Disponent **', 'Wirtschaftsinformatik, Informatik, Betriebswirtschaft, IT ** Mathematik **',
       '** Konstrukteur **',
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

lng = ['** Sehr gute Kenntnisse in Deutsch und Französisch **',
       '** Muttersprache Deutsch **', '** Fremdsprachenkenntnissen **', '** Französisch **',
       '** Englisch (sehr gut in Wort und Schrift) **']


def get_classes_to_eval(t2c, labels, indir, infiles, outdir, outfile):
    "d in only new classes to eval, and write them to outfile (if there ar new terms per class, we miss it)"
    for infile in infiles:
        with open(os.path.join(indir, infile), 'r') as f:
            for line in f:
                d = json.loads(line)
                t = get_fullterm(d)
                # print(t)
                results = d['escoQueryResults']
                for r in results[0:10]:
                    # print(r)
                    c = r['classL4up']
                    oterm = r['termEsco']
                    # print(c, oterm)
                    if t in t2c:
                        # print('jobadterm in dict')
                        if c in t2c[t]:
                            # print('class eval already:', c)
                            continue
                        # t2c[t][c].append(oterm)
                        else:
                            # print('new class:', c)
                            t2c[t][c] = [oterm]
                    else:
                        t2c[t] = {c: [oterm]}

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, outfile), 'w') as of:
        of.write('jobadterm\tclass_numeric\tclass_label\tontologyterms\n')
        for k, classes_terms in t2c.items():
            # print(labels[c]) #isced-f/0311
            for c, terms in classes_terms.items():
                # print(k, c, terms)# labels, terms)
                # of.write(k + '\t' + c + '\t' + labels[c] + '\t' + ' | '.join(set(terms)) + '\n')
                of.write(k + '\t' + c + '\t' + labels[c] + '\n')


def get_uris_to_evaluate(u2e, uri2preflabel, indir, infiles, outdir, outfile):
    "get new uris to evaluate (write everything, also old terms with annoatations to outfile)"

    uri2results = {}
    for file in infiles:
        with open(os.path.join(indir, file), 'r') as f:
            for line in f:
                d = json.loads(line)
                # print(d)
                fullterm = get_fullterm(d)
                # print(d)
                resultlist = [uri2preflabel[r['uri']] + '_' + r['classL4up'] + '_' + r['classL4up_contextLabel'] for r
                              in d['escoQueryResults'][0:10]]

                if fullterm in uri2results:
                    uri2results[fullterm] += resultlist
                else:
                    uri2results[fullterm] = resultlist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, outfile), 'w') as of:
        for k, v in uri2results.items():
            for r in set(v):
                if k in u2e and r in u2e[k]:
                    # continue
                    of.write(k + '\t' + r + '\t' + u2e[k][r] + '\n')
                else:
                    of.write(k + '\t' + r + '\n')


def gradient_selection(sims):
    "select nr of considered terms with gradient selection"
    if len(sims) == 1:
        nr_relTerms = 1
    else:
        g = np.gradient(sims)
        treshold_prob = sims[np.argmin(g)]

        sims_rev = sims[::-1]
        last_index_rev = sims_rev.index(treshold_prob)
        last_index = len(sims) - last_index_rev
        sims_rel = sims[:max(last_index, 1)]
        nr_relTerms = len(sims_rel)

    return nr_relTerms


def get_AP10(d, eval_dict, eval_level, eval_scheme, gradSel, lastIndex=10, uri2preflabel=None ):
    # ap = get_AP10(d, eval_dict, uri2preflabel, level, evalscheme, gradient_selector, lastIndex)
    "calculate average precision with cutoff value, deafult 10"

    fullterm = get_fullterm(d)

    if gradSel == True:
        esco_results = pd.DataFrame(d['escoQueryResults'])
        esco_results.sort_values(by=['similarity_score'], ascending=False, inplace=True)
        sims = esco_results.head(lastIndex)['similarity_score'].tolist()
        # sims = [r['cosineSim'] for r in d['escoQueryResults'][0:lastIndex]]  # if float(r['cosineSim']) >= 0.5]
        nr_terms = gradient_selection(sims)
        lastIndex = nr_terms

    if eval_level == 'class':
        esco_results = pd.DataFrame(d['escoQueryResults'])
        # esco_results.sort_values(by=['similarity_score'], ascending=False, inplace=True)
        resultlist = esco_results.head(lastIndex)['classL4up'].tolist()
        resultlist = [class_name.replace('http://data.europa.eu/esco/', '') for class_name in resultlist]
        # resultlist = [r['classL4up'] for r in d['escoQueryResults'][0:lastIndex]]
    # elif eval_level == 'term':
    #	resultlist = [r['termEsco'] + '_' + r['classL4up']+  '_' + r['classL4up_contextLabel'] for r in d['escoQueryResults'][0:lastIndex]]
    elif eval_level == 'uri':

        resultlist = [uri2preflabel[r['uri']] + '_' + r['classL4up'] + '_' + r['classL4up_contextLabel'] for r in
                      d['escoQueryResults'][0:lastIndex]]

    c_all = 0.0
    c_true = 0.0
    p = 0.0

    for r in resultlist:
        c_all += 1

        try:
            if r in eval_dict[fullterm]:
                tr = float(eval_dict[fullterm][r])
            # dirty fix for repr. of uris (if term euqels class label, classlabel omitted in human eval file)
            else:
                r_try = '_'.join(r.split('_')[0:2]) + '_'
                tr = float(eval_dict[fullterm][r_try])
        except:
            print(f"ontology term needs human eval., otherwise measures not meaningful: {fullterm, r}")
            tr = 0
        # try:
        # 	tr = float(t2e[fullterm][r])
        # except:
        # 	r = '_'.join(r.split('_')[0:2]) + '_'
        # 	tr = float(t2e[fullterm][r])
        if eval_scheme == 'soft':
            tr = 1 if tr >= 0.3 else 0
        elif eval_scheme == 'hard':
            tr = 1 if tr >= 0.6 else 0

        c_true += tr
        if tr > 0:
            p += c_true / c_all
    try:
        ap = p * (1 / c_true)
    except ZeroDivisionError:
        ap = 0
    return np.round(ap, 3)


def get_terminfos(d, t2e, uri2preflabel, eval_level, eval_scheme, gradSel):
    "calculate average precision with cutoff value 10"

    # print('n get terminfos')
    fullterm = get_fullterm(d)
    # print(fullterm)

    lastIndex = 10
    if gradSel == True:
        # print('gradSel')

        sims = [r['cosineSim'] for r in d['escoQueryResults'][0:10]]  # if float(r['cosineSim']) >= 0.5]
        # print(sims)
        nr_terms = gradient_selection(sims)
    # lastIndex = nr_terms

    # print('nr terms relsevant', nr_terms)
    if eval_level == 'class':
        resultlist = [r['classL4up'] for r in d['escoQueryResults'][0:lastIndex]]

    elif eval_level == 'term':
        resultlist = [r['termEsco'] + '_' + r['classL4up'] + '_' + r['classL4up_contextLabel'] for r in
                      d['escoQueryResults'][0:lastIndex]]
    elif eval_level == 'uri':
        resultlist = [uri2preflabel[r['uri']] + '_' + r['classL4up'] + '_' + r['classL4up_contextLabel'] for r in
                      d['escoQueryResults'][0:lastIndex]]

    term_results = []
    for i, r in enumerate(resultlist):
        # print(i)
        firstTerm = 1 if i == 0 else 0
        if i < nr_terms or gradSel is False:
            gradSel = 1
        else:
            gradSel = 0
        sim = sims[i]
        try:
            tr = float(t2e[fullterm][r])
        except:
            continue

        if eval_scheme == 'soft':
            tr_r = 1 if tr >= 0.3 else 0
        elif eval_scheme == 'hard':
            tr_r = 1 if tr >= 0.6 else 0
        else:
            tr_r = tr

        tr_r_soft = 1 if tr >= 0.3 else 0
        tr_r_hard = 1 if tr >= 0.6 else 0

        if eval_level == 'class':
            r += '_' + uri2preflabel[r]
        term_results.append({'i': i, 'r': r, 'sim': sim, 'eval_score': tr, 'eval_score_rnd_soft': tr_r_soft,
                             'eval_score_rnd_hard': tr_r_hard, 'gradSel': gradSel, 'firstTerm': firstTerm})
    return term_results


def get_sims(d, t2e, uri2preflabel, eval_level, eval_scheme, gradSel):
    "calculate average precision with cutoff value 10"

    fullterm = get_fullterm(d)

    lastIndex = 10
    if gradSel == True:

        sims = [r['cosineSim'] for r in d['escoQueryResults'][0:10]]  # if float(r['cosineSim']) >= 0.5]
        resultlist = [uri2preflabel[r['uri']] + '_' + r['classL4up'] + '_' + r['classL4up_contextLabel'] for r in
                      d['escoQueryResults'][0:lastIndex]]

        nr_terms = gradient_selection(sims)
        sim_TPs = []
        sim_FPs = []

        # lastIndex = nr_terms
        for s, r in zip(sims, resultlist):

            try:
                if r in t2e[fullterm]:
                    tr = float(t2e[fullterm][r])
                # dirty fix for repr. of uris (if term euqels class label, classlabel omitted in human eval file)
                else:
                    r_try = '_'.join(r.split('_')[0:2]) + '_'
                    tr = float(t2e[fullterm][r_try])
            except:
                print(f"ontology term needs human eval., otherwise measures not meaningful: {fullterm, r}")
                tr = 0

            # tr = float(t2e[fullterm][r])

            if eval_scheme == 'soft':
                tr_r = 1 if tr >= 0.3 else 0
            elif eval_scheme == 'hard':
                tr_r = 1 if tr >= 0.6 else 0
            else:
                tr_r = tr

            if tr_r == 0:
                sim_FPs.append(s)
            elif tr_r == 1:
                sim_TPs.append(s)

        sim_mean = np.mean(sims)
        sim_gradSel_mean = np.mean(sims[:nr_terms])
        sim_first = sims[0]
        sim_mean_TPs = np.nanmean(sim_TPs)
        try:
            highestSimFP = sim_FPs[0]
        except:
            highestSimFP = np.nan

    return sim_mean, sim_gradSel_mean, sim_first, sim_mean_TPs, highestSimFP


def show_terms(sample, t2c, uri2preflabel, indir, infiles, outdir, level, measure, scheme, gradSel, lastIndex):
    "get for different resultsfiles (per model) an overview of terms, sims, evaluation scores .."
    m2eval = {}
    for file in infiles[0:]:
        # get model name
        # m = file.replace('.jsonl.bz2', '').replace('terms-test-MAPPED2-terms-ontology-', '')
        m = file.replace('.jsonl.bz2', '')
        print(m)
        m2eval[m] = {}

        with open(os.path.join(indir, file), 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                fullterm = get_fullterm(d)
                # print(fullterm)
                # term = t2f[d['termJobad'].replace('\n', ' ')]
                if fullterm not in t2c:
                    print('not yet evalualted: ', fullterm)
                    continue
                results = get_terminfos(d, t2c, uri2preflabel, level, scheme, gradSel)
                # for r in results:
                #	print(r)
                m2eval[m][fullterm] = results

    for k, v in m2eval.items():
        if level == 'uri':
            outfilename = 'uris-' + k + '.tsv'
        elif level == 'class':
            outfilename = 'class-' + k + '.tsv'

        with open(os.path.join(outdir, outfilename), 'w', encoding='utf-8') as of:
            of.write(
                'model\tjobadterm\ti\tresult\tsim\teval_score\teval_score_rnd_soft\tevalscore_rnd_hard\tgradSel\tfirstTerm\n')

            for i, j in v.items():

                for d in j:
                    #	j_str = [str(l) for l in j]
                    d_str = '\t'.join([str(x) for x in d.values()])

                    # ostring = '\t'.join(j_str)
                    of.write(k + '\t' + i + '\t' + d_str + '\n')


def show_sims(sample, t2c, uri2preflabel, indir, infiles, outdir, outfile, level, measure, scheme, gradSel, lastIndex):
    "get some statistics for similarity values per model, write to file"

    m2eval = {}
    for file in infiles[0:]:
        # get model name
        # m = file.replace('.jsonl.bz2', '').replace('terms-test-MAPPED2-terms-ontology-', '')
        m = file.replace('.jsonl.bz2', '')
        # print(m)
        m2eval[m] = {}

        edumeans = []
        expmeans = []
        lngmeans = []
        allmeans = []
        with open(os.path.join(indir, file), 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                fullterm = get_fullterm(d)
                # print(fullterm)
                # term = t2f[d['termJobad'].replace('\n', ' ')]
                if fullterm not in t2c:
                    print('not yet evalualted: ', fullterm)
                    continue
                mean, meanGS, first, meanTP, highestFP = get_sims(d, t2c, uri2preflabel, level, scheme, gradSel)

                # m2eval[m][fullterm] = {'simMean': mean, 'simgradSelMean': meanGS, 'simFirst': first, 'simTPsMean': meanTP, 'simHighestFP': highestFP}

                if fullterm in edu:
                    edumeans.append([mean, meanGS, first, meanTP, highestFP])

                elif fullterm in exp:
                    expmeans.append([mean, meanGS, first, meanTP, highestFP])
                elif fullterm in lng:
                    lngmeans.append([mean, meanGS, first, meanTP, highestFP])
                allmeans.append([mean, meanGS, first, meanTP, highestFP])

        edu_v = np.nanmean(edumeans, axis=0)
        exp_v = np.nanmean(expmeans, axis=0)
        lng_v = np.nanmean(lngmeans, axis=0)
        all_v = np.nanmean(allmeans, axis=0)
        # print(all_v)
        m2eval[m]['edu'] = {'simMean': edu_v[0], 'simgradSelMean': edu_v[1], 'simFirst': edu_v[2],
                            'simTPsMean': edu_v[3], 'simHighestFP': edu_v[4]}
        m2eval[m]['exp'] = {'simMean': exp_v[0], 'simgradSelMean': exp_v[1], 'simFirst': exp_v[2],
                            'simTPsMean': exp_v[3], 'simHighestFP': exp_v[4]}
        m2eval[m]['lng'] = {'simMean': lng_v[0], 'simgradSelMean': lng_v[1], 'simFirst': lng_v[2],
                            'simTPsMean': lng_v[3], 'simHighestFP': lng_v[4]}
        m2eval[m]['all'] = {'simMean': all_v[0], 'simgradSelMean': all_v[1], 'simFirst': all_v[2],
                            'simTPsMean': all_v[3], 'simHighestFP': all_v[4]}

    df_lists = []
    for k, v in m2eval.items():
        d = {'model': k}
        for i, j in v.items():
            for u, v in j.items():
                # d[i] = j
                d[i + '_' + u] = v
        df_lists.append(d)
    df = pd.DataFrame(df_lists)
    # print(df)
    # print(df.columns.values)
    outfile_tsv = os.path.join(outdir, outfile.replace('.jsonl', '.tsv'))
    # df = pd.read_json(os.path.join(outdir, outfile), orient='records', lines=True, encoding='utf-8')
    df.to_csv(outfile_tsv, sep='\t', index=False, encoding='utf-8')


def evaluate(sample, eval_dict,  query_results_dir, infiles, outdir, outfile, level, measure, evalscheme,
             gradient_selector, lastIndex, uri2preflabel=None):
    "get for all query results from infiles, AP per term and mAP, write to outfile"
    # print('we are in evalute')
    m2eval = {}
    # print(infiles)
    for file in infiles[0:]:
        # get model name
        #	print(file)
        m = file.replace('.jsonl', '').split('/')[-2]  # .replace('terms-test-MAPPED2-terms-ontology-', '')

        print(m)
        m2eval[m] = {}
        #	print('measure:', measure)
        with open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                fullterm = get_fullterm(d)
                #		print(fullterm)
                # term = t2f[d['termJobad'].replace('\n', ' ')]
                if fullterm not in eval_dict:
                    #		print('not yet evalualted: ', fullterm)
                    continue

                if measure == 'mAP':
                    # get average precision
                    ap = get_AP10(d, eval_dict, level, evalscheme, gradient_selector, lastIndex, uri2preflabel)
                    m2eval[m][fullterm] = ap
            # print(ap)
        # elif measure == 'prec':
        #	p = get_precision(d, t2c, uri2preflabel, level, scheme, gradSel, lastIndex)
        # print(p)
        #	m2eval[m][fullterm] = p

        if sample == 'random':
            values = [v for k, v in m2eval[m].items() if k in edu]
            m2eval[m][measure + '_edu'] = np.round(np.mean(values), 3)

            values = [v for k, v in m2eval[m].items() if k in exp]
            m2eval[m][measure + '_exp'] = np.round(np.mean(values), 3)

            values = [v for k, v in m2eval[m].items() if k in lng]
            m2eval[m][measure + '_lng'] = np.round(np.mean(values), 3)

            all = edu + exp + lng

            values = [v for k, v in m2eval[m].items() if k in all]
            m2eval[m][measure] = np.round(np.mean(values), 3)
        elif sample == 'challenge':
            values = [v for k, v in m2eval[m].items()]
            m2eval[m][measure] = np.round(np.mean(values), 3)

    df_lists = []
    for k, v in m2eval.items():
        d = {'model': k}
        # print(d)
        for i, j in v.items():
            d[i] = j
        df_lists.append(d)
    df = pd.DataFrame(df_lists)
    # print(df)

    if sample == 'random':
        for i in ['_edu', '_exp', '_lng']:
            df['rank' + i] = df[measure + i].rank(method='min', ascending=False).astype(int)

    df['rank_all'] = df[measure].rank(method='min', ascending=False).astype(int)

    cols = list(df.columns.values)
    measures = [c for c in cols if c.startswith('pre') or c.startswith('mA')]
    ranks = [c for c in cols if c.startswith('rank')]
    cols_sorted = ['model'] + measures + ranks
    rest = [c for c in cols if c not in cols_sorted]
    cols_sorted += rest
    df = df[cols_sorted]
    df.sort_values(by='rank_all', inplace=True)
    outfile_tsv = os.path.join(outdir, outfile.replace('.jsonl', '.tsv'))
    df.to_csv(outfile_tsv, sep='\t', index=False, encoding='utf-8')


def get_class_annotations(infiles):
    "read in fom file wiht evaluation of classes, last column as value (if not yet mean, agg before) return a dictionary that gives for each term and each class a value between 0 and 1"
    term2classannotation = {}
    for infile in infiles:
        with open(infile, 'rt', encoding='utf-8') as f:
            for line in f:
                # skip first lite
                if line.startswith('jobadterm'):
                    continue
                line = line.strip().split('\t')
                if 'jobadterm' in line:
                    continue
                fullterm = line[1].strip()
                classuri = line[2]

                rater_average_value = float(line[-1])
                if fullterm in term2classannotation:
                    term2classannotation[fullterm][classuri] = rater_average_value
                else:
                    term2classannotation[fullterm] = {classuri: rater_average_value}

    return term2classannotation


def combine_dicts(uridict, termdict):
    "combine annotations for uri and term"
    outdict = {}
    for k, v in uridict.items():
        outdict[k] = v
    for i, j in termdict.items():
        for term in j:
            if term not in outdict[i]:
                outdict[i][term] = j[term]
    return outdict


def main():
    language = 'de'
    for sample in ['challenge']:
        if sample == 'random':

            # labels = get_classlabels('../skillontology/ESCO/classL4up_labels.jsonl')
            # uri2pref = get_uripreflabels('skillontology/terms-ontology-de-cleaned-ap-dedup.jsonl')

            # for regular evaluation of random sample
            infiles = [
                f'final_annotations/random_{language}.tsv']

            term_class_with_rater_mean_score = get_class_annotations(infiles)
            queryresults_dir = f'new_results_nearest_terms/{language}_nearest_neighbours'
            all_model_dirs = os.listdir(queryresults_dir)
            outdir = f'new_random_eval_results_{language}'

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            all_random_files = []
            for directory in all_model_dirs:
                directory_files = os.listdir(os.path.join(queryresults_dir, directory))
                if 'random-sample-with-context.jsonl' in directory_files:
                    all_random_files.append(os.path.join(queryresults_dir, directory,
                                                         'random-sample-with-context.jsonl'))

            print(all_random_files)


            measure = 'mAP'
            # levels = ['uri', 'class']
            levels = ['class']
            # evaldicts = [term_uri_rater_mean_score, term_class_with_rater_mean_score]
            evaldicts = [term_class_with_rater_mean_score]
            eval_scheme = ['soft', 'hard']
            lastIndex = [1, 5]
            gradSel = [False, False]
            for level, eval_dict in zip(levels, evaldicts):
                for evalscheme in eval_scheme:
                    for last_index, gradient_selector in zip(lastIndex, gradSel):
                        outfile = 'eval-' + sample + '-' + level + '-' + measure
                        outfile += '@' + str(last_index)
                        outfile += '-' + evalscheme
                        outfile += '-gradSel' if gradient_selector is True else ''
                        outfile += '.jsonl'
                        print(outfile)
                        evaluate(sample, eval_dict, queryresults_dir, all_random_files, outdir, outfile, level, measure,
                                 evalscheme, gradient_selector,
                                 last_index)

            # # #get an overview of resulte per model (for each queried term the results, with their sims etc on class and urilevel)
            show_terms(sample, term_uri_rater_mean_score, uri2pref, queryresults_dir, infiles, outdir, 'uri', measure,
                       'soft', True, 10)
            show_terms(sample, term_class_with_rater_mean_score, uri2pref, queryresults_dir, infiles, outdir, 'class',
                       measure, 'soft', True, 10)
            # #
            show_sims(sample, term_uri_rater_mean_score, uri2pref, queryresults_dir, infiles, outdir, 'sims_soft.tsv',
                      'uri', measure, 'soft',
                      True, 10)
            show_sims(sample, term_uri_rater_mean_score, uri2pref, queryresults_dir, infiles, outdir, 'sims_hard.tsv',
                      'uri', measure, 'hard',
                      True, 10)

        if sample == 'challenge':

            # read in annotations by one ore more annoators (classbased)
            infiles = [
                # f'final_annotations/challenge_set_de.tsv'
                f'final_annotations/challenge_set_{language}.tsv'
            ]
            term_class_with_rater_mean_score = get_class_annotations(infiles)

            # queryresults_dir = 'queryresults_random_sample_test'
            # outdir = 'evaluation-random-sample_test-2022-11-10'

            # get evaluations:
            queryresults_dir = f'new_results_nearest_terms/{language}_nearest_neighbours'
            all_model_dirs = os.listdir(queryresults_dir)

            all_challenge_files = []
            for directory in all_model_dirs:
                directory_files = os.listdir(os.path.join(queryresults_dir, directory))
                if 'challenge-sample-with-context.jsonl' in directory_files:
                    all_challenge_files.append(os.path.join(queryresults_dir, directory,
                                                            'challenge-sample-with-context.jsonl'))

            print(all_challenge_files)
            outdir = f'new_challenge_eval_results_{language}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # get mAP,  with diffeent configurations (class and uri-level, hard and soft eval, map@1, map@10, and with/withoug grad sel.)
            measure = 'mAP'
            levels = ['class']
            evaldicts = [term_class_with_rater_mean_score]
            eval_scheme = ['soft', 'hard']
            gradSel = [False, False]
            lastIndex = [1, 5]
            for level, eval_dict in zip(levels, evaldicts):
                for evalscheme in eval_scheme:
                    for last_index, gradient_selector in zip(lastIndex, gradSel):
                        outfile = 'eval-' + sample + '-' + level + '-' + measure
                        outfile += '@' + str(last_index)
                        outfile += '-' + evalscheme
                        outfile += '-gradSel' if gradient_selector is True else ''
                        outfile += '.jsonl'
                        evaluate(sample, eval_dict, queryresults_dir, all_challenge_files, outdir, outfile, level, measure,
                                 evalscheme, gradient_selector,
                                 last_index)

            show_terms(sample, term_class_with_rater_mean_score, uri2pref, queryresults_dir, infiles, outdir, 'class',
                       measure, 'soft', True, 10)


if __name__ == '__main__':
    main()
