from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc
from sqlalchemy.sql import text, exists
from sqlalchemy_utils import database_exists, create_database, drop_database
from annoy import AnnoyIndex
import spacy
import string
import pickle
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from datetime import datetime
import json


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
        self.process_input = lambda x: self.pca.transform(self.vectorizer.transform([spacy_tokenizer(x)]).toarray())[0]

    def __call__(self, text, n=10):
        # returns list of indices of nearest neighbors
        return self.nn.get_nns_by_vector(self.process_input(text), n, search_k=-1, include_distances=False)


Base = declarative_base() 

class DatabaseObject():
    # some helpful functions    
    def get(self,key='None'):
        if hasattr(self, key):
            return getattr(self,key)
        else:
            raise Exception("no key: {}".format(key))

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val):
        if hasattr(self, key):
            setattr(self,key,val)
        else:
            raise Exception("no key: {}".format(key))
            
    def __repr__(self):
        string = " <" + str(type(self).__name__) + "> " 
        for k in self.keys():
            string += "{}:{}, ".format(k,self[k])
        return string[:-2]
    
    def toJSON(self):
        js = {}
        for k in self.keys():
            js[k] = self[k]
        return js

class Database():

    def __init__(self, settings=None, dtype=None, SearchFunction=None):
        self.settings = settings
        self.dtype = dtype
        self.SearchFunction = SearchFunction
        self.create_session()
    
    def create_session(self):
        self.engine = create_engine(self.engine_string)
        Base.metadata.bind = self.engine
        self.DBSession = sessionmaker(bind=self.engine)
        self.sess = self.DBSession()

    def insert(self, data):
        self.session.add(data)
        self.session.commit()

    @property
    def count(self):
        return self.session.query(self.dtype).count()
    
    @property
    def engine_string(self):
        # local db
        if self.settings['dialect'] == 'postgresql':
            mystring = "{}://{}:{}@{}:{}/{}".format(
                self.settings['dialect'],
                self.settings['username'],
                self.settings['password'],
                self.settings['endpoint'],
                self.settings['port'],
                self.settings['dbname'] )
        elif self.settings['dialect'] == 'sqlite':
            mystring = "{}:///{}".format(
                self.settings['dialect'],
                self.settings['dbname'] )
        return mystring

    def _check_session(foo):
        def magic( self ) :
            try:
                return foo( self )
            except:
                self.close()
                del self.engine, self.DBSession, self.sess
                self.create_session()
                return foo( self )
                # TODO create something that quits after N fails? 
        return magic

    def remove(self, *args):
        self.session.query(self.dtype).filter(*args).delete()

    def query(self, *args, count=10):
        return self.session.query(self.dtype).filter(*args).limit(count).all()

    def search(self, text, count=10):
        """
        Nearest neighbor search
        """
        ids = self.SearchFunction(text)
        # get entries from database
        # entrys = db.session.query(PaperEntry.title,PaperEntry.abstract,PaperEntry.bibcode).filter(PaperEntry.id.in_(vecs)).all()
        return self.session.query(self.dtype).filter(self.dtype.id.in_(ids)).all()

    @property
    @_check_session
    def session(self):
        return self.sess

    def __del__(self):
        try:
            print('closing sessions')
            self.session.close()
        except:
            pass 

    def exists(self,key,val):
        return self.session.query(self.dtype).filter(key==val).first()

    def create_read_only_user(self,table):
        with self.engine.begin() as con:
            #con.execute(text("""GRANT USAGE ON SCHEMA public TO read_only;"""))
            #con.execute(text("""GRANT SELECT ON ALL TABLES IN SCHEMA public TO read_only;"""))
            #con.execute(text("""ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO read_only;"""))
            con.execute(text("""DROP USER read_only;"""))
            con.execute(text("""CREATE USER read_only WITH PASSWORD 'secret';"""))
            con.execute(text("""GRANT CONNECT ON DATABASE cryptocurrency TO read_only;"""))
            con.execute(text("""GRANT USAGE ON SCHEMA public TO read_only;"""))
            con.execute(text("""GRANT SELECT ON coinbasepro TO read_only;""")) # TODO use input argument for table
    
    def close(self):
        self.session.close()

    @staticmethod
    def load(filename, dtype, nearest_neighbor_search=False):
        # load database settings from json file
        with open(filename,'r') as f:
            settings = json.load(f)

        # default to false cus loading takes a min each time
        if nearest_neighbor_search:
            nnsearch = NearestNeighborSearch(settings['vectorizer_path'], settings['pca_path'], settings['annoy_path'])
        else:
            nnsearch = None

        return Database(settings=settings['database'], dtype=dtype, SearchFunction=nnsearch)


class PaperEntry(Base, DatabaseObject):
    __tablename__ = "papers"

    # define columns of table
    id = Column(Integer, autoincrement=True)
    bibcode = Column(String, primary_key=True)
    bibtex = Column(String)
    title = Column(String)
    abstract = Column(String)    # abstract text
    vec = Column(String)         # vectorized abstract
    text = Column(String)        # either full text or tokenized abstract
    categories = Column(String)  # list of categories
    pub = Column(String)
    year = Column(Integer)
    doi = Column(String)

    @staticmethod
    def keys():
        return ['id', 'bibcode', 'bibtex', 'title', 'abstract', \
                'vec', 'pub', 'year', 'categories', 'doi']

##############################

if __name__ == "__main__":
    # migrate table
    # dbOLD = Database( settings=settings['database'], dtype=OldEntry)
    # dbNEW = Database( settings=settings['database'], dtype=ADSEntry)
    # # create new table and migrate old entries
    # ADSEntry.__table__.drop(dbNEW.engine)
    # ADSEntry.__table__.create(dbNEW.engine)
    # # migrate old entries to new table
    # entrys = dbOLD.session.query(OldEntry).all()
    # for i in tqdm(range(len(entrys))):
    #     data = entrys[i].toJSON()
    #     data['id'] = i
    #     dbNEW.session.add(ADSEntry(**data))
    # dbNEW.session.commit()
    # dude()

    # create new table
    dbNEW = Database.load('settings.json', dtype=PaperEntry)

    if not database_exists(dbNEW.engine.url):
        create_database(dbNEW.engine.url)
    else:
        drop_database(dbNEW.engine.url)
        print("dropped")
        create_database(dbNEW.engine.url)
        print("created")

    print("checking existence:", database_exists(dbNEW.engine.url))

    dbNEW.dtype.__table__.create(dbNEW.engine)
    print("Number of entries:",dbNEW.count)
    #dtype.__table__.drop(dbNEW.engine)

    '''
    for i in range(5):
        data = {
                'timestamp':datetime.utcnow(),
                'latitude': np.random.random()*10+20, 
                'longitude': -120 + np.random.random()*15,
                'user':'The dude',
                'url':'https://poly.google.com/view/6FrJ3_CzH8S',
                'api':'google poly'
        }

        dbTEST.session.add( ARTest(**data) )
        dbTEST.session.commit()
    '''