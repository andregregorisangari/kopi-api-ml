import os
import numpy as np
import tensorflow as tf
from http import HTTPStatus
from PIL import Image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.layers import DepthwiseConv2D
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_KONIRA_CLASSIFICATION'] = 'models/model-konira.h5'


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' argument
        super().__init__(*args, **kwargs)


# Register the custom layer
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

model_classification = tf.keras.models.load_model(app.config['MODEL_KONIRA_CLASSIFICATION'], custom_objects=custom_objects, compile=False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'Message': 'KONIRA Apps',
    }), HTTPStatus.OK


@app.route('/predict', methods=['POST'])
def predict_konira_classification():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'No file part in the request'
                }
            }), HTTPStatus.BAD_REQUEST
        
        reqImage = request.files['image']
        
        if reqImage.filename == '':
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'No file selected for uploading'
                }
            }), HTTPStatus.BAD_REQUEST
        
        if reqImage and allowed_file(reqImage.filename):
            filename = secure_filename(reqImage.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            reqImage.save(file_path)
            
            # Load and preprocess the image
            img = Image.open(file_path).convert('RGB')
            img = img.resize((224, 224))  # Resize if necessary
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0  # Normalize pixel values
            
            # Predict with the loaded model
            prediction = model_classification.predict(x)
            class_list = ['miner', 'modisease', 'phoma', 'rust']
            predicted_class_idx = np.argmax(prediction[0])
            classification_class = class_list[predicted_class_idx]
            accuracy = float(prediction[0][predicted_class_idx]) * 100.0
            
            # Determine classtype based on classification_class (for example purposes)
            classtype = 'disease' if classification_class in ['modisease', 'phoma', 'rust'] else 'crop'
            
            # Remove uploaded image file after prediction
            os.remove(file_path)
            
            return jsonify({
                'status': {
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': {
                        'class': classification_class,
                        'accuracy': round(accuracy, 2)  # Round accuracy to two decimal places
                    }
                }
            }), HTTPStatus.OK
        else:
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'
                }
            }), HTTPStatus.BAD_REQUEST
    else:
        return jsonify({
            'status': {
                'code': HTTPStatus.METHOD_NOT_ALLOWED,
                'message': 'Method not allowed'
            }
        }), HTTPStatus.METHOD_NOT_ALLOWED


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
