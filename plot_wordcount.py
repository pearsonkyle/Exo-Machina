import os
import ads
import json
import argparse
import numpy as np
from wordcloud import WordCloud

import matplotlib.pyplot as plt

from database import Database, ADSEntry

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Settings file"
    parser.add_argument("-s", "--settings", help=help_, default="settings.json", type=str)

    help_ = "Settings key"
    parser.add_argument("-k", "--key", help=help_, default="database", type=str)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    ADSDatabase = Database( settings=settings[args.key], dtype=ADSEntry )
    
    years = ADSDatabase.session.query(ADSEntry.year).all()
    years = np.array(years).T[0]
    plt.hist(years,bins=np.arange(min(years),max(years)+1))
    plt.xlabel("Year")
    plt.grid(True,ls='--')
    plt.tight_layout()
    plt.xlim([np.mean(years)-3*np.std(years),max(years)])
    plt.savefig("year_histogram.pdf")
    plt.show()

    # citation counts
    # counts = ADSDatabase.session.query(ADSEntry.citation_count).all()

    publications = ADSDatabase.session.query(ADSEntry.pub).all()
    pubs,counts = np.unique(publications, return_counts=True)
    si = np.argsort(counts)[::-1]
    for i in range(15):
        print(" {} - {}".format(counts[si[i]], pubs[si[i]]))

    titles = ADSDatabase.session.query(ADSEntry.title).all()
    allwords = []
    for tit in titles:
        if tit[0]:
            if 'SUP' in tit[0] or 'SUB' in tit[0]:
                continue
            
            allwords.extend(tit[0].split(' '))

    keywords = ADSDatabase.session.query(ADSEntry.keyword).all()
    for tit in titles:
        if tit[0]:
            if 'SUP' in tit[0] or 'SUB' in tit[0]:
                continue
            allwords.extend(tit[0].split(' '))

    # # Generate a word cloud image
    titlecloud = WordCloud(width=1600, height=800).generate(' '.join(allwords))

    f,ax = plt.subplots(1,figsize=(14,8))
    ax.imshow(titlecloud)
    ax.axis("off")
    plt.tight_layout()
    f.savefig("wordcloud.pdf",dpi=1000)
    plt.show()
