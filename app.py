
#importing required libraries
from flask import Flask, render_template, request, send_from_directory
from flask_ngrok import run_with_ngrok
from fastai.vision import *
import os


#creating the flask object
app = Flask(__name__)
run_with_ngrok(app)

app.config["UPLOAD_FOLDER"] = 'uploads/'


#loading the model
predictor = load_learner("/content/")


#creating a predictor function
#takes image path as parameter and returns the predicted class
def my_predictor(img):
  test_img =open_image(img)
  predict_class,pred_idx,outputs=predictor.predict(test_img)
  return predict_class


#creating routes
@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  # predicting images
  imagefile = request.files['imagefile']
  filename = imagefile.filename
  image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  imagefile.save(image_path)
  
  result = my_predictor(image_path)
  
  return render_template('index.html', imagefile = imagefile.filename, prediction_text='Image is of a {}'.format(result))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__=='__main__':
  app.run()