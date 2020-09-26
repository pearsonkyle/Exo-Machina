import sys
import json
from functools import wraps

from flask import Flask, request, Response

from database import Database
from database import ARTest as DataType

settings = json.load(open("settings.json",'r'))

app = Flask(__name__)

# build response or report error 
def response(f):    
    @wraps(f)
    def wrapper(*args,**kwargs):
        db = Database( settings=settings['database'], dtype=DataType )
    
        # if we get an error, handle it
        try:
            data, message = f( db=db, *args, **kwargs )
        except Exception as e:
            data = []
            if hasattr(e,'message'):
                message = f.__name__ +'() >> '+type(e).__name__ + " : " + e.message      
            else:
                message = f.__name__ +'() >> '+type(e).__name__ + " : " + e.args[0]

        # build response + close SQL connection   
        db.close(); del db
        values = {'data':[i.toJSON() for i in data], 'message':message}
        resp = Response( json.dumps(values) ) 
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    return wrapper

@app.route('/mostrecent/<int:num>',methods=['GET'])
@response
def mostrecent(num=10,**kwargs):
    db = kwargs['db']
    return db.session.query(db.dtype).order_by(db.dtype.timestamp.desc()).limit(num).all(),'success'

# http://localhost:5000/boxselect/0.01?latitude=32&longitude=-110
@app.route('/boxselect/<float:box>',methods=['GET'])
@response
def box_select(box=0.01,**kwargs):
    db = kwargs['db']
    kwargs = request.args.to_dict()
    qlist = []
    for k in kwargs.keys():
        qlist.append( getattr(DataType,k) > (float(kwargs[k])-box) )
        qlist.append( getattr(DataType,k) < (float(kwargs[k])+box) )
    data = db.query(*qlist, count=10) # TODO add count as url argument? 
    return data,'success'

# TODO add normal select like ?user=test

@app.route('/insert',methods=['GET','POST'])
@response
def insert(db):
    if request.method == 'POST': # TODO test
        data = request.form.to_dict(flat=False)
    elif request.method == 'GET':
        data = request.args.to_dict()

    missing = []
    for k in db.dtype.keys():
        if k not in data.keys():
            missing.append(k) 

    if missing:
        return [],"missing columns: "+','.join(missing)
    else:
        db.session.add( db.dtype(**data) )
        db.session.commit()
        return [db.dtype(**data)],'success'

@app.route('/delete',methods=['GET'])
@response
def delete(db):
    count_before = db.count
    data = request.args.to_dict()
    qlist = []
    for k in data.keys():
        qlist.append( getattr(DataType,k) == data[k] ) # only works for strings?
    db.session.query(DataType).filter(*qlist).delete()
    db.session.commit()
    count_after = db.count
    return [],'success, {} rows deleted'.format(count_before-count_after)

@app.route('/delete_all',methods=['GET'])
@response
def delete_all(db):
    num = db.session.query(db.dtype).delete()
    db.session.commit()
    return [],'success, {} rows deleted'.format(num)
    
@app.route('/count',methods=['GET'])
@response
def count(db):
    return [], 'success, {} entries in db'.format(db.count)

@app.route("/my_ip", methods=["GET"])
def my_ip():
    return request.remote_addr
    
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return 'hello world' # TODO use map box to display locations and links to api


if __name__ == "__main__":

    #import pdb; pdb.set_trace()
    try:
        print(' running on port =',sys.argv[1] )
        app.run(host='0.0.0.0',port=sys.argv[1])
    except:
        app.run(host='0.0.0.0',debug=True)