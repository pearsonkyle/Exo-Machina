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

from database import Database, PaperEntry

# use tokenizer from spacy trained on sci corpus
spacy.prefer_gpu()
parser = spacy.load("en_core_sci_sm",disable=["ner"])
parser.max_length = 7000000
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
spacy.prefer_gpu()

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

    help_ = "Tokenize text"
    parser.add_argument("-t", "--tokenize", help=help_, action='store_true')

    return parser.parse_args()

class NearestNeighborSearch():
    """
    A class to perform nearest neighbor search on a vectorized text corpus. The pipeline includes 
    tokenization, tfidf vectorization, PCA dimensionality reduction, and cosine similarity.
    """

    def __init__(self, vectorizer_path, pca_path, nn_path ):
        self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        self.pca = pickle.load(open(pca_path, "rb"))
        self.nn = AnnoyIndex(self.pca.n_components_, 'angular')
        self.nn.load(nn_path)
        # lambda function to process input
        self.process_input = lambda x: pca.transform(vectorizer.transform([spacy_tokenizer(x)]).toarray())[0]

    def __call__(self, text):
        # returns list of indices of nearest neighbors
        return self.nn.get_nns_by_vector(self.process_input(text), 10, search_k=-1, include_distances=False)

    def train(self, text): # TODO finish training fn
        # vectorize text
        X = self.vectorizer.fit_transform(text)
        # reduce dimensionality
        X = self.pca.fit_transform(X.toarray())
        # build index
        for i, v in enumerate(X):
            self.nn.add_item(i, v)
        self.nn.build(10)

        # save vectorizer and pca
        pickle.dump(self.vectorizer, open("vectorizer.pkl", "wb"))
        pickle.dump(self.pca, open("pca.pkl", "wb"))
        # save index
        self.nn.save('index.ann')

if __name__ == '__main__':

    args = parse_args()

    # load database
    db = Database.load('settings.json', dtype=PaperEntry)
    print('querying database...')

    # get all abstracts
    entrys = db.session.query(PaperEntry.title,PaperEntry.abstract,PaperEntry.text,PaperEntry.bibcode).all()
    abstracts = [entry.abstract for entry in entrys]
    # read in text column and avoid reprocessing
    processed_abstracts = [entry.text for entry in entrys]

    if args.tokenize:

        # tokenize text
        for i in tqdm(range(len(abstracts))):
            # process if need be, takes ~40 min for 100k abstracts
            try:
                processed_abstracts.append( spacy_tokenizer(abstracts[i]))
                # update id + text in database
                db.session.query(PaperEntry).filter(PaperEntry.bibcode == entrys[i].bibcode).update({
                    PaperEntry.id: len(processed_abstracts)-1,
                    PaperEntry.text: processed_abstracts[-1]
                })
            except:
                # delete from databsae
                #db.session.query(PaperEntry).filter(PaperEntry.bibcode == entrys[i].bibcode).delete()
                print('deleted entry:',entrys[i].bibcode,entrys[i].abstract)
                import pdb; pdb.set_trace()
                continue

        # commit to db
        db.session.commit()
    else:
        # update id in database
        for i in tqdm(range(len(abstracts))):
            db.session.query(PaperEntry).filter(PaperEntry.bibcode == entrys[i].bibcode).update({
                PaperEntry.id: i,
            })
        # commit to db
        db.session.commit()

    # ranks words by importance/occurence
    # The maximum number of features will be the maximum number of unique words
    vectorizer = TfidfVectorizer(max_features=2**13) # too few features and brain coral talk on mars gets correlated with ocean coral talk on earth
    X = vectorizer.fit_transform(processed_abstracts) 
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("before pca:",X.shape)

    # reduce dimensionality
    pca = PCA(n_components=0.6, random_state=42)
    X_reduced= pca.fit_transform(X.toarray())
    pickle.dump(pca, open(f"pca_{X_reduced.shape[1]:d}.pkl", "wb"))

    # lambda function to process input
    process_input = lambda x: pca.transform(vectorizer.transform([spacy_tokenizer(x)]).toarray())[0]

    # compare shapes
    print("after pca:",X_reduced.shape)

    # build nearest neighbor index
    t = AnnoyIndex(X_reduced.shape[1], 'angular')
    for i in range(len(X_reduced)):
        t.add_item(i, X_reduced[i])
    t.build(10) # 10 trees
    t.save(f'tfdif+pca_{X_reduced.shape[1]:d}.ann')

    # test query
    u = AnnoyIndex(X_reduced.shape[1], 'angular')
    u.load(f'tfdif+pca_{X_reduced.shape[1]:d}.ann') # super fast, will just mmap the file
    #print(u.get_nns_by_item(0, 10)) # will find the 1000 nearest neighbors

    custom_text = "One of the key drivers of the Mars Exploration Program is the search for evidence of past or present life. In this context, the most relevant martian environments to search for extant life are those associated with liquid water, and locations that experience periodic thawing of near-surface ice. In this work, we use convolutional neural networks to detect surface regions containing Brain Coral terrain, a landform on Mars whose similarity in morphology and scale to sorted stone circles on Earth suggests that it may have formed as a consequence of freeze/thaw cycles."
    vecs = u.get_nns_by_vector(process_input(custom_text), 10, search_k=-1, include_distances=False)
    # query database for multiple ids
    entrys = db.session.query(PaperEntry.title,PaperEntry.abstract,PaperEntry.bibcode).filter(PaperEntry.id.in_(vecs)).all()
    for entry in entrys:
        print(f"{entry.title} ({entry.bibcode}) {entry.abstract}\n")

    # for i in vecs:
    #     entry = db.session.query(PaperEntry).filter(PaperEntry.id == i).first()
    #     print(entry.bibcode,entry.title,entry.abstract)
    # TODO move all this to database.py?