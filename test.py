from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import numpy as np
import scipy.io
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Setting upload folder to a temporary directory supported in Render
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mat'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set file size limit to 16MB
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the AI model (ensure the 'AERO.h5' is in your deployment's working directory)
model = tf.keras.models.load_model('AERO.h5')

def allowed_file(filename):
    """Check if the uploaded file is allowed based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_class(mat_file_path):
    """Process the .mat file and make a prediction using the model."""
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_file_path)

    # Adjust key if necessary ('data' key here must match the structure of your .mat file)
    data = mat_contents.get('data')
    if data is None:
        raise ValueError("Key 'data' not found in the .mat file. Please check the file structure.")

    # Preprocess the data (normalizing)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Define sequence length expected by the model
    sequence_length = 100  # Change this according to your model's input shape

    # Reshape data to fit the model's input requirements
    if len(data) >= sequence_length:
        reshaped_data = data[:sequence_length].reshape(1, sequence_length, data.shape[1])
    else:
        raise ValueError(f"Data length is less than the expected sequence length. Expected: {sequence_length}, Got: {len(data)}")

    # Make prediction
    prediction = model.predict(reshaped_data)

    # Debug: Print the raw prediction output (optional)
    print("Raw prediction output:", prediction)

    # Classes (modify based on the output classes of your model)
    classes = ["bird", "bird+mini-helicopter", "3_blades_long", "RC-Plane", "3_blades_short", "drone_mat"]
    predicted_class = classes[np.argmax(prediction)]

    return predicted_class

# HTML form route (Home page)
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
                # Predict the class using the uploaded .mat file
                result = predict_class(file_path)
            except ValueError as e:
                flash(str(e))
                return redirect(request.url)

    return render_template('aero.html', result=result)

# API route for prediction (REST API)
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
            # Predict the class using the uploaded .mat file
            result = predict_class(file_path)
            return jsonify({'prediction': result}), 200  # Return JSON response
        except ValueError as e:
            return jsonify({'error': str(e)}), 400  # Return JSON response

    return jsonify({'error': 'File not allowed'}), 400  # Return JSON response

# Entry point
if __name__ == '__main__':
    # Use dynamic port provided by Render (or default to port 5000 for local testing)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
