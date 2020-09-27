import os
import ads
import json
import argparse
import numpy as np

from database import Database, ADSEntry
from plot_wordcount import check_in

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
    print('querying database...')
    entrys = ADSDatabase.session.query(ADSEntry.title,ADSEntry.abstract).all()
    abstracts = []
    for entry in entrys:
        title,abstract = entry

        bad_words = [
            'galaxy','galaxies','dark matter',
            'dark energy','quasar','black hole',
            'cosmology','Black Hole', 'Cosmology',
            'Galaxy','Globular','globular','cluster',
            'Cluster','Quasar','Dark Energy', 'cosmological',
            'NGC','Herbig','Galaxies'
        ]

        if title and abstract:
            bmask = check_in(title,bad_words)
            if not bmask:
                abstracts.append(abstract)

    print("Total Entries:",len(entrys))
    print("Filtered Entries:",len(abstracts))
    with open("abstracts.txt","w") as afile:
        for i in range(len(abstracts)):

            abstr = abstracts[i]
            if not abstr:
                continue
            abstr = abstr.replace('~','about ')
            abstr = abstr.replace("<SUB>","")
            abstr = abstr.replace("</SUB>","")
            abstr = abstr.replace("<SUP>","")
            abstr = abstr.replace("</SUP>","")
            abstr = abstr.replace("<BR /> ","")
            abstr = abstr.replace(" <P />","")
            abstr = abstr.replace(" <A />","")
            abstr = abstr.replace("{","")
            abstr = abstr.replace("}","")
            # find a way to remove urls

            afile.write(abstr+"\n")
