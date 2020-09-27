from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc
from sqlalchemy.sql import text, exists

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

    @property
    def count(self):
        return self.session.query(self.dtype).count()
        
    @property
    def engine_string(self):
        # local db
        mystring = "{}://{}:{}@{}:{}/{}".format(
            self.settings['dialect'],
            self.settings['username'],
            self.settings['password'],
            self.settings['endpoint'],
            self.settings['port'],
            self.settings['dbname'] )
        return mystring

    def _check_session(foo):
        def magic( self ) :
            try:
                return foo( self )
            except:
                self.create_session()
                return foo( self )
                # TODO create something that quits after N fails? 
        return magic

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

################### custom table
class ADSEntry(Base, DatabaseObject):
    __tablename__ = "exoplanet"

    @staticmethod
    def keys():
        return ['bibcode', 'title', 'citation_count', 'abstract', \
        'pub', 'year', 'keyword','text']

    # define columns of table
    bibcode = Column(String, primary_key=True)
    title = Column(String)
    citation_count = Column(Integer)
    abstract = Column(String)
    pub = Column(String)
    year = Column(Integer)
    keyword = Column(String)
    text = Column(String)
##############################

if __name__ == "__main__":
    settings = json.load(open('settings.json', 'r'))
    dbTEST = Database( settings=settings['database'], dtype=ADS_DB )

    #ARTest.__table__.create(dbTEST.engine)
    
    mostrecent = dbTEST.session.query(ARTest).order_by(ARTest.timestamp.desc()).first()
    print(mostrecent)
    results = dbTEST.query(ARTest.latitude<32,ARTest.longitude>-110, count=10)
    print(results)

    import pdb; pdb.set_trace() 
    #dbTEST.session.query(ARTest).filter(ARTest.latitude<32).limit(10).all()
    dbTEST.session.query(dbTEST.dtype).delete()
    dbTEST.session.commit()

    #db.session.query(dbTEST.dtype).filter(dbTEST.dtype.id==123).delete()
    #db.session.commit()

    '''
    for i in range(5):
        data = {
                'timestamp':datetime.utcnow(),
                'latitude': np.random.random()*10+20, 
                'longitude': -120 + np.random.random()*15,
                'user':'Professor Munchies',
                'url':'https://poly.google.com/view/6FrJ3_CzH8S',
                'api':'google poly'
        }

        dbTEST.session.add( ARTest(**data) )
        dbTEST.session.commit()
    '''