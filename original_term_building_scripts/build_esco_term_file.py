import datetime
import json
import logging

from smart_open import open

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    filename='logfiles/build-term-file-' + datetime.datetime.today().strftime("%Y-%m-%d") + '.log',
                    filemode='w')


def uriformat2termformat(infile, outfile):
    logger.info(f' infile {infile}, outfile, {outfile}')

    sortedList = ['term', 'termOriginal', 'termType', 'uri', 'prefLabel', 'altLabels', 'hiddenLabels', 'description',
                  'reuseLevel',
                  'essentialFor', 'optionalFor', 'relatedEssential', 'relatedOptional',
                  'narrower', 'broader', 'broaderTransitive',
                  'inSchemes', 'hierarchyLevel',
                  'source', 'escoVersion', 'creationDate']

    c_a = 0
    c_h = 0
    with open(infile, 'r') as f:
        with open(outfile, 'w') as of:
            for line in f:
                d = json.loads(line.strip())

                # entry for prefLabel
                d['term'] = shorten_terms(d['prefLabel'])
                d['termOriginal'] = d['prefLabel']

                d['hierarchyLevel'] = get_level(d['uri'])
                # if   d['hierarchyLevel']  == 0:
                #    print(d['uri'])
                d['termType'] = 'prefLabel'
                d_new = {e: d[e] for e in sortedList}
                of.write(json.dumps(d_new, ensure_ascii=False) + '\n')

                for a in d['altLabels']:
                    c_a += 1
                    d['term'] = shorten_terms(a)
                    d['termOriginal'] = a
                    d['termType'] = 'altLabel'

                    d_new = {e: d[e] for e in sortedList}
                    of.write(json.dumps(d_new, ensure_ascii=False) + '\n')

                for h in d['hiddenLabels']:
                    c_h += 1
                    d['term'] = shorten_terms(h)
                    d['termOriginal'] = h
                    d['termType'] = 'hiddenLabel'
                    d_new = {e: d[e] for e in sortedList}
                    of.write(json.dumps(d_new, ensure_ascii=False) + '\n')
    logger.info(
        ' converted format: one-line per uri to one-line per term (with new attributes term, termOriginal, termType. term is a cleaned version of termOriginal)')
    logger.info(f' additional entries for alt labels: {c_a}')
    logger.info(f' additional entries for hidden labels: {c_h}')


def shorten_terms(text):
    text = text.replace(' – nirgendwo sonst klassifiziert', '')
    text = text.replace(' – nicht näher definiert', '')
    text = text.replace(' (ohne Sprachen)', '')
    return text


def get_level(uri):
    if uri == 'skill':
        level = 0
    elif uri.startswith('skill'):
        if len(uri.split('.')) == 3:
            level = 4  # skill/S14.5.6
        elif len(uri.split('.')) == 2:
            level = 3  # skill/S1.2
        elif len(uri.split('.')) == 1:
            if '/' in uri and uri.split('/')[1].isalpha():  # #skill/S, skill/T, skill/L, skill/2
                level = 1
            elif '-' in uri:  # skill/1973c966-f236-40c9-b2d4-5d71a89019be
                level = 5
            else:  # skill/S1
                level = 2

    elif uri.startswith('isced'):
        level = len(uri.split('/')[1])
    return level


def get_lowest_above(levels, maxLevel):
    lowest = 0
    for l in levels:
        if l > lowest and l < maxLevel:
            lowest = l
    return lowest


def get_graph(escofile, graphfile):
    graph = {}
    graph.setdefault('skill', []).append('skill/L')
    graph.setdefault('skill', []).append('skill/T')
    graph.setdefault('skill', []).append('skill/K')
    graph.setdefault('skill', []).append('skill/S')

    with open(escofile, 'r') as f:
        for line in f:
            d = json.loads(line)
            if d['termType'] != 'prefLabel':
                continue
            child = d['uri']
            for parent in d['broaderTransitive']: #todo: this has to be broader term
                graph.setdefault(parent, []).append(child)

    with open(graphfile, 'w') as of:
        of.write(json.dumps(graph))
    logger.info(f' graph containing all hierarchical relations written to file: {graphfile}')
    return graph


def find_all_paths(graph, start, end, path=[]):
    # paths = find_all_paths(g, 'skill', d['uri'], path=[])
    path = path + [start]
    if start == end:
        return [path]

    if not start in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def write_all_paths(infile, g, outfile):
    sortedList = ['term', 'termOriginal', 'termType', 'uri', 'prefLabel', 'altLabels', 'hiddenLabels', 'description',
                  'reuseLevel',
                  'essentialFor', 'optionalFor', 'relatedEssential', 'relatedOptional', 'inSchemes',
                  'broaderTransitive', 'hierarchyLevel', 'classL4up',
                  'source', 'escoVersion', 'creationDate']
    c_e = 0
    c_p = 0
    with open(infile, 'r') as f:
        with open(outfile, 'w') as of:
            for line in f:
                c_e += 1
                d = json.loads(line.strip())
                # print(line)
                # print(d['prefLabel'])
                ##term, termType, uri, prefLabel, altLabels, hiddenLabels, description, scope, reuseLevel, types, topOf, "inSchemes": ["skills", "member-skills"], "creation_date": "2021-11-29", "esco_version": "v.1.0.8", "hierarchyLevel": 5}
                paths = find_all_paths(g, 'skill', d['uri'], path=[])
                for p in paths:
                    c_p += 1
                    d_new = {k: v for k, v in d.items() if k not in ['broader', 'broaderTransitive', 'narrower']}

                    if int(d['hierarchyLevel']) < 5:
                        cL4up = d['uri']
                    else:
                        trans_levels = [get_level(b) for b in p]
                        lowest_level = get_lowest_above(trans_levels, 5)
                        ll_index = trans_levels.index(lowest_level)  # and prob here too
                        cL4up = p[ll_index]
                    d_new['classL4up'] = cL4up
                    d_new['broaderTransitive'] = p
                    # d_new['source'] = 'esco'
                    # d_new['escoVersion'] = d['esco_version']
                    # d_new['creationDate'] =  datetime.datetime.today().strftime("%Y-%m-%d")
                    d_out = {e: d_new[e] for e in sortedList}
                    of.write(json.dumps(d_out, ensure_ascii=False) + '\n')
    logger.info(f' written new outfile, with a separate entry for every upward path {outfile}')
    logger.info(f' original {c_e} entries, new {c_p} entries')


def add_sjmm_terms(infile_esco, infile_sjmm, outfile):
    with open(outfile, 'w') as of:
        with open(infile_esco, 'r') as f:
            for line in f:
                of.write(line)
        with open(infile_sjmm, 'r') as f:
            for line in f:
                of.write(line)


def main():
    # urifile = 'ESCO/v1.1.0/uris-esco-de.jsonl'
    termfile = 'skillontology/terms-esco-sjmm-skills-de-cleaned-ap-dedup.jsonl'
    # uriformat2termformat(urifile, termfile)
    #
    graphfile = 'skillontology/graph_hierarchy.json'
    g = get_graph(termfile, graphfile)
    print(g)

    # infile = 'ESCO/v1.1.0/terms-esco-orig-de-cleaned.jsonl'
    # outfile = 'ESCO/v1.1.0/terms-esco-orig-de-cleaned-ap.jsonl'
    # write_all_paths(infile, g, outfile)

    # for additions


    # infile = 'additions_esco/terms_sjmm_additions_complete.jsonl'
    outfile = 'skillontology/terms-sjmm-additions-esco-ap.jsonl'
    infile = 'skillontology/terms-esco-sjmm-skills-de-cleaned-ap-dedup.jsonl'
    write_all_paths(infile, g, outfile)

    # add files togheter
    infile_esco = 'ESCO/v1.1.0/terms-esco-orig-de-cleaned-ap.jsonl'
    infile_sjmm = 'additions_esco/terms-sjmm-additions-esco-ap.jsonl'
    outfile = 'ESCO/v1.1.0/terms-esco-de-cleaned-ap.jsonl'
    add_sjmm_terms(infile_esco, infile_sjmm, outfile)

if __name__ == '__main__':
    main()
