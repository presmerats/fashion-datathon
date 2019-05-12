from flask import Flask, jsonify, render_template, request
import time
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)

@app.route("/")
def main():
    with open('server/data/demo.json') as json_file:
        data = json.load(json_file)

    data2 = json.dumps(data, ensure_ascii=False)
    print(data['people'])
    print(type(data['people']))
    return render_template('main.html', my_data=data['people'])



@app.route("/table")
def add():
    with open('server/data/demo.json') as json_file:
        data = json.load(json_file)

    return json.dumps(data, ensure_ascii=False)
"""
with open('server/data/demo.json') as json_file:
    data = json.load(json_file)
    print(data) # tal cual el archivo



    render_template('template.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])
"""