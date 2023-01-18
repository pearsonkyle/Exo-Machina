import os
import json
import random
import argparse
from tqdm import tqdm

from database import Database
from database import ARXIVEntry as Entry

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Settings file"
    parser.add_argument("-s", "--settings", help=help_, default="settings.json", type=str)

    help_ = "Output file"
    parser.add_argument("-o", "--output", help=help_, default="abstracts.txt", type=str)

    help_ = "Settings key"
    parser.add_argument("-k", "--key", help=help_, default="database", type=str)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    DB = Database( settings=settings[args.key], dtype=Entry )
    print(f'querying database... ({DB.count} entries)')
    entrys = DB.session.query(Entry.title,Entry.abstract,Entry.bibcode).order_by(Entry.id).all()

    # randomize the order of the abstracts
    random.shuffle(entrys)
    #entrys = entrys[:int(len(entrys)/2)]
    #entrys = entrys[int(len(entrys)/2):]

    # open output file
    with open(args.output, 'w', errors='ignore') as f:
        # loop over lines and write abstracts
        for i in tqdm(range(len(entrys))):
            title,abstract,bibcode = entrys[i]

            # clean up the text
            abstr = abstract.replace('~','about ')
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
            abstr = abstr.replace(b"\xc3\xa2\xc2\x80\xc2", b" ")
            abstr = abstr.replace(b"\xc3\xa2\xc2\x80\xc2\x99", b"")
            abstr = abstr.replace(b"\\times",b"x")
            abstr = abstr.replace(b"\xc2\xb5m",b"micron")
            abstr = abstr.replace(b"\x99",b"'")

            abstr = abstr.replace(b"\xc2\xa9",b"")
            abstr = abstr.replace(b"\xcb\x87",b"")
            abstr = abstr.replace(b"\xe2\x88\x92",b"-")
            abstr = abstr.replace(b"\xef\xac\x82",b"fl")
            abstr = abstr.replace(b"\xce\xb1",b"-alpha-")
            abstr = abstr.replace(b"\xc3\x97",b"~")
            abstr = abstr.replace(b"\xe2\x8a\x99",b"_sun")
            abstr = abstr.replace(b"\xe2\x80\x93",b"-")

            abstr = abstr.replace(b"\x93",b"")
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
            abstr = abstr.replace(b"\xbf",b"") 
            abstr = abstr.replace(b"\xc3\xa2\xc2\x88\xc2\x9e",b"")
            abstr = abstr.replace(b"\xc3\xa2\xc2\x89\xc2\xa4",b"")
            abstr = abstr.replace(b"\xc2",b"")
            abstr = abstr.replace(b"\t",b"")

            abstr = abstr.replace(b"     ",b" ")
            abstr = abstr.replace(b"  ",b" ")

            # write abstract to file
            if len(abstr)>100:
                f.write(abstr.decode("utf-8") + '\n')
                # replace entry in database with cleaned up version
                DB.session.query(Entry).filter(Entry.bibcode==bibcode).update({Entry.abstract:abstr.decode("utf-8")})

    # commit changes to database
    DB.session.commit()