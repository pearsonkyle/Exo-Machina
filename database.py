from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, desc
from sqlalchemy.sql import text, exists
from sqlalchemy_utils import database_exists, create_database, drop_database
from annoy import AnnoyIndex
import spacy
import string
import pickle
#python -m spacy download en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from datetime import datetime
import json


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
    bibtex = Column(String)               # bibtex entry
    title = Column(String)
    pub = Column(String)
    year = Column(Integer)
    abstract = Column(String)             # abstract text
    abstract_embedding = Column(String)   # abstract embedding
    abstract_processed = Column(String)   # tokenized abstract for tfidf
    text = Column(String)                 # full text
    text_embedding = Column(String)       # full text embedding
    text_processed = Column(String)       # tokenized full text for tfidf
    categories = Column(String)           # list of categories

    @staticmethod
    def keys():
        return ['id', 'bibcode', 'bibtex', 'title', 'pub', 'year', 'categories', \
                'abstract', 'abstract_embedding', 'abstract_processed', \
                'text', 'text_embedding', 'text_processed']

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
    # example of using sqlalchemy to add to database

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