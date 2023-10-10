import os
import ads
import argparse

from database import Database, PaperEntry

ads.config.token = os.environ.get('ADS_DEV_KEY')

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Initial search criteria"
    parser.add_argument("-q", "--query", help=help_, default="exoplanet", type=str)

    help_ = "Settings file"
    parser.add_argument("-s", "--settings", help=help_, default="settings.json", type=str)

    help_ = "Max pages"
    parser.add_argument("-m", "--max_pages", help=help_, default=15, type=int)

    help_ = "Loosen search criteria, query text doesn't have to be in abstract"
    parser.add_argument("-l", "--loose_query", help=help_, action='store_true')

    help_ = "Sort based on (score, citation_count, year, bibcode)"
    parser.add_argument("-o", "--sort", help=help_, default="score", type=str)

    return parser.parse_args()

def format_entry(response, query):

    # create entry
    data = {}
    for k in ['bibcode','title','abstract','pub','year']:
        val = getattr(response,k)
        if isinstance(val,list):
            data[k] = str(val[0])
        else:
            data[k] = val

    # format
    data['year'] = int(data['year'])
    data['categories'] = query
    return data


if __name__ == '__main__':

    args = parse_args()

    #settings = json.load(open(args.settings, 'r'))
    #ADSDatabase = Database( settings=settings["database"], dtype=ADSEntry )
    db = Database.load('settings.json', dtype=PaperEntry)

    # initial query
    papers = ads.SearchQuery(
        q=args.query, 
        fl=[
            'title', 'abstract', 'bibtex',
            'pub', 'year', 'keyword','bibcode'
        ],
        sort=args.sort, max_pages=args.max_pages
    )

    bibcodes = []
    # add papers to db
    for paper in papers:
        data = format_entry(paper, args.query)

        # skip if abstract is none
        if data['abstract'] is None:
            continue

        # check for query text in abstract
        if not args.loose_query and args.query.lower() not in data['abstract'].lower():
            continue

        # check that value doesn't exist
        checkval = db.exists(PaperEntry.bibcode,paper.bibcode)
        if not checkval:
            db.insert( PaperEntry(**data))
            print(data['title'], data['bibcode'])
            bibcodes.append(data['bibcode'])

    # query db for all bibcodes
    #bibcodes = db.session.query(PaperEntry.bibcode).order_by(-PaperEntry.year).all()
    print('Total DB Entries:', db.count)

    # get each papers references
    for bibcode in bibcodes:
        papers = ads.SearchQuery(
            q="references(bibcode:{})".format(bibcode), 
            fl=[
                'title', 'abstract', 'bibtex',
                'pub', 'year', 'keyword', 'bibcode'
            ],
            sort=args.sort, max_pages=args.max_pages
        )

        # add papers to db
        for paper in papers:
            data = format_entry(paper, args.query)

            # skip if abstract is none
            if data['abstract'] is None:
                continue

            # check that value doesn't exist
            checkval = db.exists(PaperEntry.bibcode,paper.bibcode)
            if not checkval: 
                db.session.add( PaperEntry(**data))
                db.session.commit()
                print(paper.title, paper.bibcode)

    # TODO have it clean up the special character, gt, lt, sup, sub, etc.
    # TODO add another script to query Exoplanet.EU for planet references

    '''fields
    paper.abstract              paper.build_citation_tree   paper.first_author_norm     paper.keys                  paper.pubdate
    paper.aff                   paper.build_reference_tree  paper.id                    paper.keyword               paper.read_count
    paper.author                paper.citation              paper.identifier            paper.metrics               paper.reference
    paper.bibcode               paper.citation_count        paper.issue                 paper.page                  paper.title
    paper.bibstem               paper.database              paper.items                 paper.property              paper.volume
    paper.bibtex                paper.first_author          paper.iteritems             paper.pub                   paper.year
    '''