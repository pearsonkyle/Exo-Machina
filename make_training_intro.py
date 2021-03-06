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

    help_ = "Delete bad keys"
    parser.add_argument("-d","--delete", help=help_, default=False)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    ADSDatabase = Database( settings=settings[args.key], dtype=ADSEntry )
    print('querying database...')
    #entrys = ADSDatabase.session.query(ADSEntry.title,ADSEntry.abstract,ADSEntry.text,ADSEntry.bibcode).filter(ADSEntry.text!="").all()
    entrys = ADSDatabase.session.query(ADSEntry.title,ADSEntry.abstract,ADSEntry.text,ADSEntry.bibcode).all()

    samples = []
    for entry in entrys:
        title,abstract,text,bibcode = entry

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
            'Super Nova', 'pulsar', 'Pulsar', 'open clusters', 'baryon', 'cosmological'
            #'Binary','Comet', 'comet', 'asteroid', 'Asteroid'
        ]

        if title and abstract:
            bmask = check_in(title,bad_words) | check_in(abstract,bad_words) | check_in(text,bad_words) #| (len(text) < 500) | (" " not in text)
            if not bmask:
                samples.append(abstract)
                if len(text) > 100:
                    samples.append(text)
            else:
                # delete bad entry
                ADSDatabase.session.query(ADSEntry).filter(ADSEntry.bibcode==bibcode).delete()
        else:
            ADSDatabase.session.query(ADSEntry).filter(ADSEntry.bibcode==bibcode).delete()
        
    # commit all the deletions
    ADSDatabase.session.commit()

    print("Total Entries:",len(entrys))
    print("Filtered Entries:",len(samples))
    with open("abstracts.txt","w") as afile:
        for i in range(len(samples)):

            if not samples[i]:
                continue

            abstr = samples[i]
            if isinstance(abstr,str):
                pass
            else:
                abstr = samples.decode("utf-8")
            
            if 'ABSTRACT' in abstr:
                abstr = abstr.split('ABSTRACT')[1]
    
            if 'Abstract' in abstr:
                abstr = abstr.split('Abstract')[1]

            if 'INTRODUCTION' in abstr:
                abstr = abstr.split('INTRODUCTION')[1]
    
            abstr = abstr.replace('~','about ')
            abstr = abstr.replace("<SUB>","_")
            abstr = abstr.replace("</SUB>","")
            abstr = abstr.replace("<SUP>","^")
            abstr = abstr.replace("</SUP>","")
            abstr = abstr.replace("<BR /> ","")
            abstr = abstr.replace(" <P />","")
            abstr = abstr.replace(" <A />","")
            abstr = abstr.replace("{","")
            abstr = abstr.replace("}","")
            abstr = abstr.replace("\t"," ")
            abstr = abstr.replace("\r"," ")

            abstr = abstr.encode("utf-8", "ignore")
            abstr = abstr.replace(b"\xef\xac\x81",b"fi")
            abstr = abstr.replace(b"\xef\xac\x83",b"fi")
            abstr = abstr.replace(b"\xe2\x80\x99",b"'")
            abstr = abstr.replace(b"\xef\xac\x80",b"ff")
            abstr = abstr.replace(b"\xe2\x88\xbc",b"~")
            abstr = abstr.replace(b"\xce\xbcm",b"micron")
            abstr = abstr.replace(b"\xce\xbc m",b"micron")
            
            abstr = abstr.replace(b"\xc2\xb5m",b"micron")

            abstr = abstr.replace(b"\xc2\xa9",b"")
            abstr = abstr.replace(b"\xcb\x87",b"")
            abstr = abstr.replace(b"\xe2\x88\x92",b"-")
            abstr = abstr.replace(b"\xef\xac\x82",b"fl")
            abstr = abstr.replace(b"\xce\xb1",b"-alpha-")
            abstr = abstr.replace(b"\xc3\x97",b"~")
            abstr = abstr.replace(b"\xe2\x8a\x99",b"_sun")
            abstr = abstr.replace(b"\xe2\x80\x93",b"-")

            abstr = abstr.replace(b"\xa0",b" ")
            abstr = abstr.replace(b"\xe2\x80",b"")
            abstr = abstr.replace(b"\x9c",b"'")
            abstr = abstr.replace(b"\x9d",b"'")
            abstr = abstr.replace(b"\x94",b"")
            abstr = abstr.replace(b"\x98",b"'")
            abstr = abstr.replace(b"\xb2",b"")
            abstr = abstr.replace(b"\xc2\xb1 ",b"+-")
            abstr = abstr.replace(b"\xce",b"")

            abstr = abstr.replace(b"\xb7",b"")
            abstr = abstr.replace(b"\xc2\xa7",b"Section")
            abstr = abstr.replace(b"\xe2\x88",b"")
            abstr = abstr.replace(b"\x97",b"")
            abstr = abstr.replace(b"\xbb",b"")

            abstr = abstr.replace(b"\t",b"")

            abstr = abstr.replace(b"     ",b" ")
            #abstr = abstr.replace(b"",b"")
            #abstr = abstr.replace(b"",b"")

            if " " not in abstr.decode("utf-8","ignore"):
                continue

            # for j in range(500,len(abstr),500):
            #     substr = abstr[j-500:j].decode("utf-8","ignore").strip()

            #     substr = ". ".join(substr.split('. ')[1:])
                
            #     if substr == "":
            #         continue

            #     if substr[0] in ['0','1','2','3','4','5','6','7','8','9']:
            #         substr = ". ".join(substr.split('. ')[1:])

            #     if substr == "":
            #         continue

            #     if substr[0] in ['0','1','2','3','4','5','6','7','8','9']:
            #         substr = "). ".join(substr.split('). ')[1:])

            #     if substr == "":
            #         continue

            #     if substr[0] == "(":
            #         substr = ")".join(substr.split(')')[1:])

            if len(abstr)>100:
                afile.write(abstr.decode("utf-8","ignore").strip()+"\n")