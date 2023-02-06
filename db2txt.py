import random
import argparse
from tqdm import tqdm
import ftfy
from cleantext import clean

from database import Database, PAPERentry

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

    # load database
    DB = Database.load('settings.json', dtype=PAPERentry)

    print(f'querying database... ({DB.count} entries)')
    entrys = DB.session.query(PAPERentry.title,PAPERentry.abstract,PAPERentry.bibcode).all()

    # randomize the order of the abstracts
    random.shuffle(entrys)

    exceptions = []

    # open output file
    with open(args.output, 'w') as f:
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

            abstr = abstr.encode("utf-8")
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
            abstr = abstr.replace(b"\xb3",b"")
            abstr = abstr.replace(b"\xb0",b"")
            abstr = abstr.replace(b"\xf0",b"")
            abstr = abstr.replace(b"\xe2\x8a\x95",b"Earth") # earth cross
            abstr = abstr.replace(b"\xe2\x8a",b"Sun") # sun dot
            abstr = abstr.replace(b"\x9e",b"Infinity")
            abstr = abstr.replace(b"\xe2\x86\x92",b"approaches")
            abstr = abstr.replace(b"\xbd",b"v")
            abstr = abstr.replace(b"\xc3\xb8",b"oi")
            abstr = abstr.replace(b"\xbd",b"nu")
            abstr = abstr.replace(b"\xb4",b"gamma")
            abstr = abstr.replace(b"\xe2\x89\x88",b"about")
            abstr = abstr.replace(b"\xe2\x89",b"greater than or equal to")
            abstr = abstr.replace(b"\xbc", b"micro ")
            abstr = abstr.replace(b"\xb4",b"delta")


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
            abstr = abstr.replace(b"\xf1",b"")
            abstr = abstr.replace(b"\xb7",b"")
            abstr = abstr.replace(b"\xc2\xa7",b"Section")
            abstr = abstr.replace(b"\xe2\x88",b"")
            abstr = abstr.replace(b"\x97",b"")
            abstr = abstr.replace(b"\xbb",b"")
            abstr = abstr.replace(b"\xbf",b"") 
            abstr = abstr.replace(b"\xc3\xa2\xc2\x88\xc2\x9e",b"")
            abstr = abstr.replace(b"\xc3\xa2\xc2\x89\xc2\xa4",b"")
            abstr = abstr.replace(b"\xc2",b"")
            abstr = abstr.replace(b"\xef",b"")
            abstr = abstr.replace(b"\xe9",b"")
            abstr = abstr.replace(b"\xed",b"")
            abstr = abstr.replace(b"\xc9",b"")

            abstr = abstr.replace(b"\t",b"")
            abstr = abstr.replace(b"     ",b" ")
            abstr = abstr.replace(b"  ",b" ")

            # remove all non utf-8 characters
            abstr = abstr.decode("utf-8", "ignore").encode("utf-8")

            abstr = ftfy.fix_text(abstr.decode("utf-8"))

            abstr = clean(abstr)

            # write abstract to file
            if len(abstr)>100:
                try:
                    f.write(abstr + '\n')

                    # replace entry in database with cleaned up version
                    #DB.session.query(PAPERentry).filter(PAPERentry.bibcode==bibcode).update({PAPERentry.abstract:abstr})
                except Exception as ex:
                    continue
                #    exceptions.append((bibcode, ex))

                    # delete entry from database
                    #DB.session.query(PAPERentry).filter(PAPERentry.bibcode==bibcode).delete()

    print(f"Exceptions: {len(exceptions)}")

    # commit changes to database
    #DB.session.commit()
    DB.close()