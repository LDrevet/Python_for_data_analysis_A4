import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("/Users/luciedrevet/Documents/S7/predict sales/mymodel")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    final_features = np.reshape(final_features,(1,6))
    prediction = model.predict_classes(final_features)
    prediction = prediction + 1921

    return render_template('index.html', prediction_text='Release year should be  {}'.format(prediction))
    #return str(final_features.shape)


if __name__ == "__main__":
    app.run(debug=True)
