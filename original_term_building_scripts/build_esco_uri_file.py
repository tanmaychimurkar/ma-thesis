import json
from smart_open import open
import logging
import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    filename='logfiles/build-uri-file-' + datetime.datetime.today().strftime("%Y-%m-%d") + '.log',
                    filemode='w')


def readin_uris(uri_files):
    uri2infos = {}
    for file in uri_files:
        logger.info(f' read uris from infile: {file}')

        with open(file, 'r') as f:
            for line in f:
                if line.startswith('?a'):
                    continue
                # print(line)
                l = line.strip().split('\t')
                uri = l[0]
                prefLabel = l[1]
                try:
                    description = l[2]
                except:
                    description = ''
                try:
                    reuseLevel = l[3]
                except:
                    reuseLevel = ''
                if uri in uri2infos:
                    logger.warning(f" uri already in dict: {l}, {uri2infos[uri]}")
                else:
                    uri2infos[uri] = {'uri': uri, 'prefLabel': prefLabel, 'description': description,
                                      'reuseLevel': reuseLevel}

    logger.info(f' {len(uri2infos.keys())} uris got read in')
    return uri2infos


def get_uri2labels(labelfile):
    """
    A function to get the alternate and the hidden labels for a specific URI. This function takes in the location of
    the tsv file, and from it derives the hidden and the altenate terms. The tsv files are the results of the queries
    that are run in SPARQL on the turtle graph.

    """
    logger.info(f' read additional infos from infile: {labelfile}')

    uri2labels = {}
    with open(labelfile, 'r') as f:
        for line in f:
            if line.startswith('?a'):
                continue
            l = line.strip().split('\t')
            uri = l[0]
            label = l[1]
            if uri in uri2labels:
                uri2labels[uri].append(label)
            else:
                uri2labels[uri] = [label]

    nr_labels = sum([len(v) for k, v in uri2labels.items()])

    # this number includes labels for occupations.
    logger.info(f' {nr_labels} entries got read in, for {len(uri2labels.keys())} uris ')
    return uri2labels


def add_labels(uri2infos, altLabelfile, hiddenLabelfile):
    """
    Function to add the alternate and the hidden lables to the collection of the following 3 terms that come for a URI:

    PrefLabel, Description, ReuseLabel

    If there is no alternate or hidden label available for a URI, then it is kept as blank.
    """
    altLabels = get_uri2labels(altLabelfile)
    hiddenLabels = get_uri2labels(hiddenLabelfile)
    c_a = 0
    c_h = 0
    for k, v in uri2infos.items():
        # check for altlabels
        if k in altLabels:
            v['altLabels'] = altLabels[k]
            c_a += 1
        else:
            v['altLabels'] = []

        # check for hiddenLabels
        if k in hiddenLabels:
            v['hiddenLabels'] = hiddenLabels[k]
            c_h += 1
        else:
            v['hiddenLabels'] = []
    logger.info(f' added alt labels for {c_a} uris ')
    logger.info(f' added hidden labels for {c_h} uris ')

    return uri2infos


def add_broader_narrower(uri2infos, narrowerFile, broaderFile, broaderTFile):
    """
    Function to further add then narrower, broader and broaderTransitive URIs to the collection of URIs that already
    have the 5 following properties:

    PrefLabel, Description, ReuseLevel, AlternateLabel, Hiddenlabel.

    If no narrower, braoder or broaderTransitive URIs are found, kept as None.
    """
    narrower = get_uri2labels(narrowerFile)
    broader = get_uri2labels(broaderFile)
    broaderT = get_uri2labels(broaderTFile)
    c_n = 0
    c_b = 0
    c_bT = 0
    for k, v in uri2infos.items():

        # check for narrower
        if k in narrower:
            v['narrower'] = narrower[k]
            c_n += 1
        else:
            v['narrower'] = []

        # check for broader
        if k in broader:
            v['broader'] = broader[k]
            c_b += 1
        else:
            v['broader'] = []

        # check for broaderTransitive
        if k in broaderT:
            v['broaderTransitive'] = broaderT[k]
            c_bT += 1
        else:
            v['broaderTransitive'] = []

    logger.info(f' added narrower entries for {c_n} uris ')
    logger.info(f' added broader entries for {c_b} uris ')
    logger.info(f' added broader transitive entries for {c_bT} uris ')
    return uri2infos


def add_relations(uri2infos, essForFile, optForFile, relEssFile, relOptFile):
    essFor = get_uri2labels(essForFile)
    optFor = get_uri2labels(optForFile)
    relEss = get_uri2labels(relEssFile)
    relOpt = get_uri2labels(relOptFile)
    c_e = 0
    c_o = 0
    c_rE = 0
    c_rO = 0

    for k, v in uri2infos.items():

        if k in essFor:
            v['essentialFor'] = essFor[k]
            c_e += 1
        else:
            v['essentialFor'] = []

        if k in optFor:
            v['optionalFor'] = optFor[k]
            c_o += 1
        else:
            v['optionalFor'] = []

        if k in relEss:
            v['relatedEssential'] = relEss[k]
            c_rE += 1
        else:
            v['relatedEssential'] = []

        if k in relOpt:
            v['relatedOptional'] = relOpt[k]
            c_rO += 1
        else:
            v['relatedOptional'] = []

    logger.info(f' added essentialFor entries for {c_e} uris ')
    logger.info(f' added essentialOptional entries for {c_o} uris ')
    logger.info(f' added relatedEssential entries for {c_rE} uris ')
    logger.info(f' added relateOptional entries for {c_rO} uris ')

    return uri2infos


def add_inSchemes(uri2infos, inSchemeFile):
    schemes = get_uri2labels(inSchemeFile)
    c_s = 0

    for k, v in uri2infos.items():

        if k in schemes:
            v['inSchemes'] = schemes[k]
            c_s += 1
        else:
            v['inSchemes'] = []
    logger.info(f' added inScheme entries for {c_s} uris ')
    return uri2infos


def write2file(uri2infos, outfile):
    with open(outfile, 'w') as f:
        for k, v in uri2infos.items():
            v['uri'] = shorten_uris(v['uri'])

            v['description'] = clean_descriptions(v['description'])

            v['narrower'] = [shorten_uris(e) for e in v['narrower']]
            v['broader'] = [shorten_uris(e) for e in v['broader']]
            v['broaderTransitive'] = [shorten_uris(e) for e in v['broaderTransitive']]
            v['essentialFor'] = [shorten_uris(e) for e in v['essentialFor']]
            v['optionalFor'] = [shorten_uris(e) for e in v['optionalFor']]
            v['relatedEssential'] = [shorten_uris(e) for e in v['relatedEssential']]
            v['relatedOptional'] = [shorten_uris(e) for e in v['relatedOptional']]
            v['inSchemes'] = [shorten_uris(e) for e in v['inSchemes']]
            v['reuseLevel'] = shorten_uris(v['reuseLevel'])

            v['prefLabel'] = clean_terms(v['prefLabel'])
            v['altLabels'] = [clean_terms(e) for e in v['altLabels']]
            v['hiddenLabels'] = [clean_terms(e) for e in v['hiddenLabels']]

            v["source"] = "esco"
            v["escoVersion"] = "v1.1.0"
            v['creationDate'] = datetime.datetime.today().strftime("%Y-%m-%d")
            d = json.dumps(v, ensure_ascii=False)
            f.write(d + '\n')
        logger.info(' shortened uris ')
        logger.info(' cleaned esco terms (@de removed)')
        logger.info(' added information on source, creation date ')
        logger.info(f' written uris to outfile: {outfile} ')


def clean_terms(term):
    # "\"IKT-Kenntnisse erfassen\"@de"
    term = term.replace('@de', '')
    term = term.replace('\"', '')
    return term


def clean_descriptions(text):
    text = text.replace('\"', '')
    text = text.replace('\\n', ' ')
    text = text.strip()
    return text


def shorten_uris(uri):
    uri = uri.replace('<http://data.europa.eu/esco/skill-reuse-level/', '')
    uri = uri.replace('<http://data.europa.eu/esco/concept-scheme/', '')
    uri = uri.replace('<http://data.europa.eu/esco/', '')

    uri = uri.replace('>', '')
    return uri


def main():
    uri_files = [
        'sparql_results/skills_hier_basicinos.tsv',
        'sparql_results/skills_nonhier_basicinfos.tsv'
    ]
    uri2infos = readin_uris(uri_files)

    altLabelfile = 'sparql_results/altLabels.tsv'
    hiddenLabelfile = 'sparql_results/hiddenLabels.tsv'
    uri2infos = add_labels(uri2infos, altLabelfile, hiddenLabelfile)

    narrowerFile = 'sparql_results/skillsNarrower.tsv'
    broaderFile = 'sparql_results/skillsBroader.tsv'
    broaderTFile = 'sparql_results/skillsBroaderTransitive.tsv'
    uri2infos = add_broader_narrower(uri2infos, narrowerFile, broaderFile, broaderTFile)

    essForFile = 'sparql_results/essentialFor.tsv'
    optForFile = 'sparql_results/optionalFor.tsv'

    relEssFile = 'sparql_results/relatedEssential.tsv'
    relOptFile = 'sparql_results/relatedOptional.tsv'
    uri2infos = add_relations(uri2infos, essForFile, optForFile, relEssFile, relOptFile)

    inSchemeFile = 'sparql_results/skills_inSchemes.tsv'
    uri2infos = add_inSchemes(uri2infos, inSchemeFile)

    outfile = 'uris-esco-de.jsonl'
    write2file(uri2infos, outfile)

    # additional infos.
    # order?


if __name__ == '__main__':
    main()
