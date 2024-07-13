from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    
    # Save the file temporarily
    img_path = 'Y8.jpg'
    file.save(img_path)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    result = np.argmax(prediction[0])

    # Delete temporary file
    os.remove(img_path)

    # Return the result as JSON response
    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
