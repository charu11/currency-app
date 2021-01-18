import os
import sys
import cv2

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/TEST-CNN.h5'

# Load your own trained model
#model = tf.keras.models.load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')


def prepare(file_path):
    print("..........................file path ")
    print(file_path)
    IMGSIZE = 100
    Img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)/255
    new_array = cv2.resize(Img_array, (IMGSIZE, IMGSIZE))

    return new_array.reshape(-1, IMGSIZE, IMGSIZE, 1)




def model_predict(img, model):
    img = img.resize((100, 100))

    # Preprocessing the image
    x = image.img_to_array(img)
   # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')
    print('###################################')
    print(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        print(img)

        # Save the image to ./uploads
        img.save("./uploads/image.jpeg")

        MODEL_PATH = 'models/TEST-CNN.h5'

        # Load your own trained model
        model = tf.keras.models.load_model(MODEL_PATH)
        testing = model.predict([prepare('./uploads/image.jpeg')])
        print('testing........................................................')
        print(testing)
        result = "Undefined"

        CATEGORIES = ["Dollar", "Pound"]
        print([float(testing[0][0])])
        print([float(testing[0][1])])
        print(CATEGORIES[int(testing[0][0])])

        if float(testing[0][0]) > 0.95:
            print("One Dollar")
            result ="One Dollar"

        elif float(testing[0][0]) < 0.95 and float(testing[0][1]) < 0.05:
            print("Not Defined")
            result = "Not Defined"

        else:
            print("five Pounds")
            result =" Five Pounds"

        pred_proba = "{:.3f}".format(np.amax(testing))  # Max probability

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
