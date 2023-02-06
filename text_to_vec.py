import os
import json
import string
import argparse
import pickle
from tqdm import tqdm
from annoy import AnnoyIndex
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from database import Database, PAPERentry

# use tokenizer from spacy trained on sci corpus
parser = spacy.load("en_core_sci_sm",disable=["ner"])
parser.max_length = 7000000
punctuations = string.punctuation
stopwords = list(STOP_WORDS)

def spacy_tokenizer(sentence):
    # not good for text generation - only use for embedding
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Settings file"
    parser.add_argument("-s", "--settings", help=help_, default="settings.json", type=str)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # load database
    db = Database.load('settings.json', dtype=PAPERentry)

    # load text data
    entrys = db.session.query(PAPERentry).order_by(PAPERentry.id).all()
    abstracts = [entry.abstract for entry in entrys]
    processed_abstracts = [entry.text for entry in entrys]

    # prep data
    for i in tqdm(range(len(abstracts))):
        # process if need be, takes ~40 min for 100k abstracts
        if entrys[i].text == "":
            # tokenize text
            processed_abstracts[i] = spacy_tokenizer(abstracts[i])
            entrys[i].text = processed_abstracts[i]
        entrys[i].id = i # reset id in database

    # save new IDs to database
    PAPERentry.session.commit()

    # ranks words by importance/occurence
    # The maximum number of features will be the maximum number of unique words
    vectorizer = TfidfVectorizer(max_features=2**13) # too few features and brain coral talk on mars gets correlated with ocean coral talk on earth
    X = vectorizer.fit_transform(processed_abstracts) 
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("before pca:",X.shape)

    # reduce dimensionality
    pca = PCA(n_components=0.5, random_state=42)
    X_reduced= pca.fit_transform(X.toarray())
    pickle.dump(pca, open("pca.pkl", "wb"))

    # lambda function to process input
    process_input = lambda x: pca.transform(vectorizer.transform([spacy_tokenizer(x)]).toarray())[0]

    # compare shapes
    print("after pca:",X_reduced.shape)

    # build nearest neighbor index
    t = AnnoyIndex(X_reduced.shape[1], 'angular')
    for i in range(len(X_reduced)):
        t.add_item(i, X_reduced[i])
    t.build(10) # 10 trees
    t.save(f'test_{X_reduced.shape[1]:d}.ann')

    # test query
    u = AnnoyIndex(X_reduced.shape[1], 'angular')
    u.load(f'test_{X_reduced.shape[1]:d}.ann') # super fast, will just mmap the file
    #print(u.get_nns_by_item(0, 10)) # will find the 1000 nearest neighbors

    custom_text = "One of the key drivers of the Mars Exploration Program is the search for evidence of past or present life. In this context, the most relevant martian environments to search for extant life are those associated with liquid water, and locations that experience periodic thawing of near-surface ice. In this work, we use convolutional neural networks to detect surface regions containing Brain Coral terrain, a landform on Mars whose similarity in morphology and scale to sorted stone circles on Earth suggests that it may have formed as a consequence of freeze/thaw cycles."
    vecs = u.get_nns_by_vector(process_input(custom_text), 10, search_k=-1, include_distances=False)
    for i in vecs:
        print(abstracts[i],'\n')