from app import app
from flask import session, redirect, url_for, request, make_response,send_file
from markupsafe import escape
import json
from dotenv import load_dotenv
import os
import torch
import hnswlib
import numpy as np
import torchvision
from flask_cors import CORS, cross_origin
import datetime
import base64
import io
import cv2
from .models import encoders
from torch.utils.data import DataLoader
from .data import transform
from .db import db
config = {'size' : 224, 'batch_size' : 1, 'num_workers' : 0, 'device' : 'cuda', 'embedding_size' : 512}
feature_extractor = encoders.resnet18()
feature_extractor.eval()
p = hnswlib.Index(space = 'cosine', dim = config['embedding_size']) # possible options are l2, cosine or ip
p.load_index('cosine_normalized_index.idx')
print('hnsw loaded')
load_dotenv()

HACKER_KEY = os.getenv("SECRET_SESSION")
SECRET_KEY = os.getenv("SECRET_KEY")
app.secret_key = SECRET_KEY.encode()
cors = CORS(app, expose_headers=['Access-Control-Allow-Origin'], supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/images/<path:path>')
@cross_origin()
def images(path):
    print(path)
    #generate_img(path)
    #resp = flask.make_response(open("./images/"+path).read())
    #resp.content_type = "images/jpg"
    return send_file("images/"+path,mimetype="image/JPG")


@app.route('/', methods=['POST'])
@cross_origin()
def index():
    key = request.headers.get('Authorization')
    print(key)
    if HACKER_KEY == base64.b64decode(key.split(' ')[1]).decode():
        #image = np.array(Image.open(io.BytesIO(request.get_data())))
        image = request.get_data()
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        image = decoded
        #im = Image.fromarray(image)
        #im.save('kek.jpg')
        #image = cv2.imread('kek.jpg')
        print(image.shape)
        print('+++')
        transforms_ = transform.Compose([transform.Pad(size=(224)),
                                   transform.ToTensor(),
                                   transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        image,_,_ = transforms_(image)
        device = torch.device("cpu")
        img = image.to(device).unsqueeze(0)
        out = feature_extractor(img)
        vec = [out.detach().numpy().squeeze()]
        q_labels, q_distances = p.knn_query(vec, k = 7)
        print(q_labels[0])
        print(q_distances[0])
        tenders_arr = []
        items_arr = []
        for i in range(len(q_labels[0])):
            q_labels[0][i]+=1
            items, name = db.get_item(q_labels[0][i])
            print(q_labels[0][i])
            items['img_url'] = 'http://35.228.6.45:5000/images/'+str(items['Id'])+'.jpg'
            items['distance'] = float(q_distances[0][i])
            items_arr.append(items)
            print(items)
            tenders = db.get_tender(name)
            tenders_arr.append(tenders)
        #ВОТ ТУТ q_labels содержит индексы ближайших пикч 
        #CODE HERE
        resp = {"items": items_arr,"tenders":tenders_arr}
        return json.dumps(resp,ensure_ascii=False), 200
    resp = {"error": "Forbidden"}

    return json.dumps(resp), 403

@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        req = request.get_json()
        #if req == None:
             #req = request.get_data()
        print(req)
        if HACKER_KEY == req['password']:
            resp = make_response({"token":base64.b64encode(bytes(HACKER_KEY, 'utf-8')).decode()})
            return resp
        resp = make_response({"error": "Forbidden"})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp, 403

@app.errorhandler(404)
@cross_origin()
def not_found(error):
    resp = {"error": "Such page does not exist"}
    return json.dumps(resp), 404
