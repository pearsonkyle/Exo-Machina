from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc
from sqlalchemy.sql import text, exists
from sqlalchemy_utils import database_exists, create_database, drop_database

from datetime import datetime
import json

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

    def __init__(self, settings=None, dtype=None):
        self.settings = settings
        self.dtype = dtype
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
    
    def query(self,*args, count=10):
        return self.session.query(self.dtype).filter(*args).limit(count).all()

    @staticmethod
    def load(filename, dtype, key='database'):
        # load database settings from json file
        with open(filename,'r') as f:
            settings = json.load(f)[key]
        return Database(settings=settings, dtype=dtype)


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