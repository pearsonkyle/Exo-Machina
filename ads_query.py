import os
import ads
import json
import argparse

from database import Database, ADSEntry

ads.config.token = 'snWV2OTqoHYMxdOBQH4VzsDHYxMBrcJTXLOgTI2N'

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Initial search criteria"
    parser.add_argument("-q", "--query", help=help_, default="exoplanet", type=str)

    help_ = "Settings file"
    parser.add_argument("-s", "--settings", help=help_, default="settings.json", type=str)

    help_ = "Settings key"
    parser.add_argument("-k", "--key", help=help_, default="database", type=str)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    ADSDatabase = Database( settings=settings[args.key], dtype=ADSEntry )

    # create db on disk
    if not os.path.exists("{}.db".format(settings[args.key]['dbname'])):
        #ADSEntry.__table__.create(ADSDatabase.engine)
        pass

    # initial query 
    papers = ads.SearchQuery(
        q=args.query, 
        fl=[
            'title', 'citation_count', 'abstract', 
            'pub', 'year', 'keyword','bibcode'
        ],
        sort="citation_count", max_pages=4
    )

    # add papers to db
    for paper in papers:
        print(paper.title, paper.citation_count)

        data = {}
        for k in ADSEntry.keys():
            val = getattr(paper,k)
            if isinstance(val,list):
                data[k] = str(val[0])
            else:
                data[k] = val

        data['year'] = int(data['year'])

        ADSDatabase.session.add( ADSEntry(**data))
        import pdb; pdb.set_trace()
        ADSDatabase.session.commit()

        # val = ADSDatabase.exists('bibcode',paper.bibcode)

    # while there are still api creds do more searches
    papers.response.get_ratelimits()
    '''
    first_paper.abstract              first_paper.build_citation_tree   first_paper.first_author_norm     first_paper.keys                  first_paper.pubdate
    first_paper.aff                   first_paper.build_reference_tree  first_paper.id                    first_paper.keyword               first_paper.read_count
    first_paper.author                first_paper.citation              first_paper.identifier            first_paper.metrics               first_paper.reference
    first_paper.bibcode               first_paper.citation_count        first_paper.issue                 first_paper.page                  first_paper.title
    first_paper.bibstem               first_paper.database              first_paper.items                 first_paper.property              first_paper.volume
    first_paper.bibtex                first_paper.first_author          first_paper.iteritems             first_paper.pub                   first_paper.year
    '''