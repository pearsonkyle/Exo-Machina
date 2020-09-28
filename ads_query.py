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

def format_entry(response):

    # create entry
    data = {}
    for k in ADSEntry.keys():
        if k == 'text': continue
        val = getattr(response,k)
        if isinstance(val,list):
            data[k] = str(val[0])
        else:
            data[k] = val

    # format
    data['text'] = ""
    data['year'] = int(data['year'])
    return data


if __name__ == '__main__':

    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    ADSDatabase = Database( settings=settings[args.key], dtype=ADSEntry )

    # create table
    #ADSEntry.__table__.create(ADSDatabase.engine)
    
    # initial query 
    papers = ads.SearchQuery(
        q=args.query, 
        fl=[
            'title', 'citation_count', 'abstract', 
            'pub', 'year', 'keyword','bibcode'
        ],
        sort="citation_count", max_pages=2
    )

    # add papers to db
    for paper in papers:
        data = format_entry(paper)

        # check that value doesn't exist
        checkval = ADSDatabase.exists(ADSEntry.bibcode,paper.bibcode)
        if not checkval: 
            ADSDatabase.session.add( ADSEntry(**data))
            ADSDatabase.session.commit()
            print(paper.title, paper.bibcode)

    # get each papers references
    bibcodes = ADSDatabase.session.query(ADSEntry.bibcode).order_by(-ADSEntry.year).all()
    print('Total DB Entries:', len(bibcodes))

    for bibcode in bibcodes:
        papers = ads.SearchQuery(
            q="references(bibcode:{})".format(bibcode[0]), 
            fl=[
                'title', 'citation_count', 'abstract', 
                'pub', 'year', 'keyword','bibcode'
            ],
            sort="citation_count", max_pages=4
        )

        try:
            # add papers to db
            for paper in papers:
                data = format_entry(paper)

                # check that value doesn't exist
                checkval = ADSDatabase.exists(ADSEntry.bibcode,paper.bibcode)
                if not checkval: 
                    ADSDatabase.session.add( ADSEntry(**data))
                    ADSDatabase.session.commit()
                    print(paper.title, paper.bibcode)
        except:
            pass

    # rates = ads.RateLimits('SearchQuery')
    # r.limits['remaining'] > 100 
    # r.limits['limit']

    '''fields
    paper.abstract              paper.build_citation_tree   paper.first_author_norm     paper.keys                  paper.pubdate
    paper.aff                   paper.build_reference_tree  paper.id                    paper.keyword               paper.read_count
    paper.author                paper.citation              paper.identifier            paper.metrics               paper.reference
    paper.bibcode               paper.citation_count        paper.issue                 paper.page                  paper.title
    paper.bibstem               paper.database              paper.items                 paper.property              paper.volume
    paper.bibtex                paper.first_author          paper.iteritems             paper.pub                   paper.year
    '''
    # citations(abstract:HST)