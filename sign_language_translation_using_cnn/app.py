from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and label binarizer
model = load_model('models/sign_language_model.h5')
with open('models/sign_lang_transform.pkl', 'rb') as f:
    label_binarizer = pickle.load(f)

# Function to process the uploaded image
def convert_image_to_array(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        # Process the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_array = convert_image_to_array(image)
        
        # Make prediction
        prediction = model.predict(image_array)
        label = label_binarizer.classes_[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        return render_template('result.html', label=label, confidence=confidence)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)