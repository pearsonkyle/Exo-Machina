# download PDF + extract text from ads bibcode
import os
import argparse
import json
import glob
import gzip
import time
from bs4 import BeautifulSoup
import urllib.request
from database import Database, ADSEntry

import arxiv

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

class PdfConverter:

   def __init__(self, file_path):
       self.file_path = file_path

   def convert_pdf_to_txt(self):
       rsrcmgr = PDFResourceManager()
       retstr = StringIO()
       codec = 'utf-8'  # 'utf16','utf-8'
       laparams = LAParams()
       device = TextConverter(rsrcmgr, retstr, laparams=laparams)
       fp = open(self.file_path, 'rb')
       interpreter = PDFPageInterpreter(rsrcmgr, device)
       password = ""
       maxpages = 0
       caching = True
       pagenos = set()
       for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
           interpreter.process_page(page)
       fp.close()
       device.close()
       str = retstr.getvalue()
       retstr.close()
       return str

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Settings file"
    parser.add_argument("-s", "--settings", help=help_, default="settings.json", type=str)

    help_ = "Settings key"
    parser.add_argument("-k", "--key", help=help_, default="database", type=str)

    return parser.parse_args()


def get_links_ads(bibcode = '2016Natur.529...59S', q = 'arxiv'):
    parser = 'html.parser'  # or 'lxml' (preferred) or 'html5lib', if installed
    resp = urllib.request.urlopen("https://ui.adsabs.harvard.edu/abs/{}/abstract".format(bibcode))
    soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))

    pdfs = []
    base_url = "https://ui.adsabs.harvard.edu"
    for link in soup.find_all('a', href=True):
        if q in link['href'].lower():
            pdfs.append(base_url+link['href'])
            
    return pdfs

def check_in(title,bad_words):
    bmask = False
    for bword in bad_words:
        if bword in title:
            bmask = True
    return bmask

def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

if __name__ == "__main__":
    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    ADSDatabase = Database( settings=settings[args.key], dtype=ADSEntry )
    

    entrys = ADSDatabase.session.query(
        ADSEntry.title,ADSEntry.bibcode,ADSEntry.abstract,
        ADSEntry.pub).all()
    for entry in entrys:
        title,bibcode,abstract,pub = entry

        if pub not in ["Science","The Astrophysical Journal","Monthly Notices of the Royal Astronomical Society"]:
            continue

        dbentry = ADSDatabase.session.query(ADSEntry).filter(ADSEntry.bibcode==bibcode).first()
        if dbentry.text != "":
            continue 

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
                'Super Nova', 'pulsar', 'Pulsar', 'open clusters', 'pulsation',
                #'Binary','Comet', 'comet', 'asteroid', 'Asteroid'
        ]

        if title and abstract:
            bmask = check_in(title,bad_words) | check_in(abstract,bad_words)
            if not bmask:
                try:
                    links = get_links_ads(bibcode,q='arxiv')
                except:# HttpError:
                    print("rate limited ",time.localtime())
                    time.sleep(300)
                    continue

                if links:
                    
                    arxivid = links[0].split('/')[-1]
                    aid = arxivid.split(':')[-1]
                    
                    try:
                        # search with api
                        paper = arxiv.query(id_list=[aid])[0]
                        arxiv.download(paper) #,prefer_source_tarfile=True)
                        pdffile = glob.glob("{}*.pdf".format(aid))[0]
                    except:
                        try:
                            section = links[0].split('/')[-2].split(':')[-1]
                            url = "https://arxiv.org/pdf/{}/{}.pdf".format(section,aid)
                            pdffile = "{}.pdf".format(aid)
                            urllib.request.urlretrieve(url, pdffile)
                        except:
                            import pdb; pdb.set_trace()
                            continue

                    # parse pdf
                    pdfConverter = PdfConverter(file_path=pdffile)
                    text = pdfConverter.convert_pdf_to_txt()

                    if pub == 'Science':
                        if 'References' in text:
                            text = text.split("References")[0]

                        if '\n\n1\n\n' in text:
                            text = text.split('\n\n1\n\n')[1]

                            for i in range(20):
                                substring = '\n{}\n\n'.format(i)
                                text = "".join(text.split(substring))

                            # remove references in brackets
                            while text.find("[") > 0:
                                p1=text.find("[") # get the position of [
                                p2=text.find("]") # get the position of ]

                                if p2 < p1:
                                    text = replace_str_index(text,p2,"")
                                else:
                                    text = text.replace(text[p1:p2+1], "")
                    
                    elif pub == "The Astrophysical Journal":
                        if 'ABSTRACT\n' in text:
                            text = text.split("ABSTRACT\n")[1]

                        if 'Introduction\n' in text:
                            text = text.split("Introduction\n")[1]

                        if '\n2.' in text:
                            text = text.split("\n2.")[0]
    
                    elif pub == "Monthly Notices of the Royal Astronomical Society":
                        if 'ABSTRACT\n' in text:
                            text = text.split("ABSTRACT\n")[1]

                        if 'INTRODUCTION\n' in text:
                            text = text.split("INTRODUCTION\n")[1]

                        if '\n2 ' in text:
                            text = text.split("\n2 ")[0]
                    else:
                        os.remove(pdffile)
                        # publication type not supported yet
                        continue

                    # clean up some new line text
                    text = " ".join(text.split('\n'))
                    text = "".join(text.split("- "))
                    text = "".join(text.split("\x0c"))
                    text = text.replace("  "," ")

                    print(text)
                    os.remove(pdffile)

                    # add text to database
                    dbentry = ADSDatabase.session.query(ADSEntry).filter(ADSEntry.bibcode==bibcode).first()
                    dbentry.text = text
                    ADSDatabase.session.commit()

                    print("Values in DB:",ADSDatabase.session.query(ADSEntry).filter(ADSEntry.text!="").count())
