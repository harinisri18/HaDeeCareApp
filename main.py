import os
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from keras.models import load_model
import keras.utils as image
from PIL import Image
from keras.utils import load_img
import keras
import tensorflow as tf
UPLOAD_FOLDER = 'C:\\Users\\HP\\OneDrive\\Documents\\SHIFT\\PCOS-Predictor Final\\pcos files\\static\\model_data\\'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading Pre-trained Model ...")
model = load_model('C:\\Users\\HP\\OneDrive\\Documents\\SHIFT\\PCOS-Predictor Final\\pcos files\\static\\model.h5')
class_map = {0: "notinfected", 
            1: "infected"}

def predict_class(filename):    
    img = keras.utils.load_img(filename, target_size=(64, 64))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    images = np.vstack([x])
    Class = model.predict(images, batch_size=1)[0]
    return class_map[int(Class[0])]


@app.route("/")
def index1():
    return render_template('index1.html')

@app.route('/index1.html')
def home():
   return render_template('index1.html')

@app.route('/predict', methods=["GET", "POST"])
def upload_image():
    Class = ''
    if request.method == "POST":
        if request.files:
            image=request.files["image"]
            imagename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], image.filename))         
            Class=predict_class(UPLOAD_FOLDER + "/" + image.filename)
            return render_template("result.html", Class=Class)

if __name__ == "__main__":
    app.run(debug=True)