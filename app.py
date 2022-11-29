import os
import numpy as np
import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template


# creating instance of the class
app = Flask(__name__, template_folder='templates')

model = pickle.load(open("model.pkl", "rb"))

# to tell flask what url should trigger the function index()


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 2)
    loaded_model = pickle.load(
        open("./model.pkl", "rb"))  # load the model
    # predict the values using loded model
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
         # Simpan data fitur object dalam array dan diubah ke float 
        float_features = np.array([float(x) for x in request.form.values()])
        # Buat dalam array 2D karena model hanya menerima 2D array
        features = np.array([float_features])
        # Prediksi class object dengan model yang telah dibuat
        prediction = model.predict(features)
        # Karena class prediksi berupa int, ubah ke bentuk 
        if(prediction==np.array([0])):
            res = "Adelie"
        elif(prediction==np.array([1])):
            res = "Chinstrap"
        elif(prediction==np.array([2])):
            res = "Gentoo"
        # Mengembalikan hasil prediksi ke index.html dalam variabel result
        return render_template("result.html", prediction = "{}".format(res))

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)