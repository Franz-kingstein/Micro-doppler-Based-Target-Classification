from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import numpy as np
import scipy.io
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mat'}
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
model = tf.keras.models.load_model('AERO.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_class(mat_file_path):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_file_path)
    
    # Replace 'data' with the actual key in your .mat file if different
    data = mat_contents.get('data')
    if data is None:
        raise ValueError("Key 'data' not found in the .mat file. Please check the file structure.")
    
    # Preprocess the data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Define the sequence length
    sequence_length = 100  # Must match the sequence length used during training
    
    # Reshape the data to match model's expected input shape
    if len(data) >= sequence_length:
        reshaped_data = data[:sequence_length].reshape(1, sequence_length, data.shape[1])
    else:
        raise ValueError(f"Data length is less than the expected sequence length. Expected: {sequence_length}, Got: {len(data)}")
    
    # Make prediction
    prediction = model.predict(reshaped_data)
    
    # Print the raw prediction for debugging
    print("Raw prediction output:", prediction)
    
    # Assuming the model returns a one-hot encoded result
    classes = ["bird", "bird+mini-helicopter", "3_blades_long", 
               "RC-Plane", "3_blades_short", "drone_mat"]
    predicted_class = classes[np.argmax(prediction)]
    
    return predicted_class

# HTML form route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Predict the class
                result = predict_class(file_path)
            except ValueError as e:
                flash(str(e))
                return redirect(request.url)

    return render_template('aero.html', result=result)

# New REST API route
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Predict the class
            result = predict_class(file_path)
            return jsonify({'prediction': result}), 200  # Ensure this returns JSON
        except ValueError as e:
            return jsonify({'error': str(e)}), 400  # Ensure this returns JSON

    return jsonify({'error': 'File not allowed'}), 400  # Ensure this returns JSON


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change port to 5001 or any available port
