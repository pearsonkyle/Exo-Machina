import os
import ads
import json
import argparse
import numpy as np

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
    
    abstracts = ADSDatabase.session.query(ADSEntry.abstract).all()
    
    print(len(abstracts))
    with open("abstracts.txt","w") as afile:
        for i in range(len(abstracts)):

            abstr = abstracts[i][0]
            if not abstr:
                continue
            abstr = abstr.replace('~','about ')
            abstr = abstr.replace("<SUB>","")
            abstr = abstr.replace("</SUB>","")
            abstr = abstr.replace("<SUP>","")
            abstr = abstr.replace("</SUP>","")
            abstr = abstr.replace("<BR /> ","")
            abstr = abstr.replace(" <P />","")
            abstr = abstr.replace("{","")
            abstr = abstr.replace("}","")
            abstr = abstr.replace("]","")
            abstr = abstr.replace("[","")
            
            afile.write(abstr+"\n")
