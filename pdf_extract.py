# download PDF + extract text from ads bibcode
import argparse
import json
import glob
import gzip

from bs4 import BeautifulSoup
import urllib.request
import PyPDF2 
import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from database import Database, ADSEntry

import arxiv


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
'''
#open allows you to read the file.
pdfFileObj = open("text.pdf",'rb')
#The pdfReader variable is a readable object that will be parsed.
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
#Discerning the number of pages will allow us to parse through all the pages.
num_pages = pdfReader.numPages
count = 0
text = ""
#The while loop will read each page.
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()
#This if statement exists to check if the above library returned words. It's done because PyPDF2 cannot read scanned files.
if text != "":
   text = text
#If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text.
else:
   text = textract.process(url, method='tesseract', language='eng')
#Now we have a text variable that contains all the text derived from our PDF file. Type print(text) to see what it contains. It likely contains a lot of spaces, possibly junk such as '\n,' etc.
#Now, we will clean our text variable and return it as a list of keywords.

# strip text between title + references
text = ' '.join(text.split('\n'))
text = text.split('references')[0]
text = text.split('!!!')[-1] # separate authors from abstract, nature

# TODO remove text from figures or download from arxiv

tokens = word_tokenize(text)
#We'll create a new list that contains punctuation we wish to clean.
punctuations = ['(',')',';',':','[',']',',']
#We initialize the stopwords variable, which is a list of words like "The," "I," "and," etc. that don't hold much value as keywords.
stop_words = stopwords.words('english')
#We create a list comprehension that only returns a list of words that are NOT IN stop_words and NOT IN punctuations.
keywords = [word for word in tokens if not word in stop_words and not word in punctuations]
'''

if __name__ == "__main__":
    args = parse_args()

    settings = json.load(open(args.settings, 'r'))
    ADSDatabase = Database( settings=settings[args.key], dtype=ADSEntry )

    entrys = ADSDatabase.session.query(
        ADSEntry.title,ADSEntry.bibcode,ADSEntry.abstract).all()
    for entry in entrys:
        title,bibcode,abstract = entry

        bad_words = [
            'galaxy','galaxies','dark matter',
            'dark energy','quasar','black hole',
            'cosmology','Black Hole', 'Cosmology',
            'Galaxy','Globular','globular','cluster',
            'Cluster','Quasar','Dark Energy', 'cosmological',
            'NGC','Herbig','Galaxies','White Dwarf','white dwarf',
        ]

        if title and abstract:
            bmask = check_in(title,bad_words) | check_in(abstract,bad_words)
            if not bmask:
                links = get_links_ads(bibcode,q='arxiv')
                if links:
                    
                    arxivid = links[0].split('/')[-1]
                    aid = arxivid.split(':')[-1]
                    paper = arxiv.query(id_list=[aid])[0]
                    # Download the gzipped tar file
                    arxiv.download(paper,prefer_source_tarfile=True)
                    gzfile = glob.glob("{}*.gz".format(aid))[0]

                    # unzip
                    # find tex document
                    # find introduction
                    # strip references

                    # urllib.request.urlretrieve(url, 'text.pdf')

                    dude()


