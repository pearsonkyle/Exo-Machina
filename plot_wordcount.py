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

    return parser.parse_args()

def remove_all(x, element):
    return list(filter(lambda a: element not in a, x))
    #return list(filter((element).__ne__, a))

def check_in(title,bad_words):
    bmask = False
    for bword in bad_words:
        if bword in title:
            bmask = True
    return bmask

if __name__ == '__main__':

    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    ADSDatabase = Database( settings=settings['database'], dtype=ADSEntry )

    entrys = ADSDatabase.session.query(
        ADSEntry.title,ADSEntry.keyword,ADSEntry.year,ADSEntry.pub,ADSEntry.abstract).all()
    allwords = []; years = []; publications =[]
    for entry in entrys:
        title,keyword,year,pub,abstract = entry

        bad_words = [
            'galaxy','galaxies','dark matter',
            'dark energy','quasar','black hole',
            'cosmology','Black Hole', 'Cosmology',
            'Galaxy','Globular','globular','cluster',
            'Cluster','Quasar','Dark Energy', 'cosmological',
            'NGC','Herbig','Galaxies','White Dwarf','white dwarf',
            'cosmic','microwave','Microwave', 'Dark energy', 
            'neurtrino', 'Neutrino', 'Quark', 'quark', 'Milky Way',
            'Galactic', 'Open Cluster', 'Open cluster', 'Cosmological',
            'Baryon', 'baryon', 'Subdwarfs', 'subdwarfs',
            'Type II', 'Type I', 'type II', 'type I', 'coronal mass',
            'Prominence', 'Prominences','Coronal mass','Boyajian',
            'interstellar medium', 'IGM', 'ISM', 'href', 'url', 'Redshifts',
            '21-cm', '21 cm', 'Relativity', 'Relativistic', 'SuperNova', 'super nova',
            'Super Nova', 'pulsar', 'Pulsar', 'open clusters', 'Ga'
            #'Binary','Comet', 'comet', 'asteroid', 'Asteroid'
        ]

        if title and abstract:
            bmask = check_in(title,bad_words) | check_in(abstract,bad_words)
            if not bmask:
                allwords.extend(title.split(' '))
                years.append(year)
                publications.append(pub)
                pass

    allwords = remove_all(allwords,'SUP')
    allwords = remove_all(allwords,'using')
    allwords = remove_all(allwords,'Using')

    allwords = remove_all(allwords,'II')
    allwords = remove_all(allwords,'III')
    allwords = remove_all(allwords,'IV')

    allwords = remove_all(allwords,'SUB')
    allwords = remove_all(allwords,'New')
    allwords = remove_all(allwords,'new')
    allwords = remove_all(allwords,'first')

    allwords = remove_all(allwords,'First')
    

    # publication journal
    pubs,counts = np.unique(publications, return_counts=True)
    si = np.argsort(counts)[::-1]
    for i in range(15):
        print(" {} - {}".format(counts[si[i]], pubs[si[i]]))

    # year of publication
    print("N Articles:",len(years))
    years = np.array(years)
    plt.hist(years,bins=np.arange(min(years),max(years)+1))
    plt.xlabel("Year")
    plt.grid(True,ls='--')
    plt.tight_layout()
    plt.xlim([np.mean(years)-3*np.std(years),max(years)])
    plt.savefig("year_histogram.png")
    plt.show()

    # # Generate a word cloud image
    titlecloud = WordCloud(
        width=1242, height=2688, colormap='ocean',
        background_color='white',font_path='Nasa21.ttf',
        normalize_plurals=False).generate(' '.join(allwords))

    f,ax = plt.subplots(1,figsize=(14,9))
    ax.imshow(titlecloud)
    ax.axis("off")
    plt.tight_layout()
    f.savefig("wordcloud.png",dpi=1000)
    plt.show()
