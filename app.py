import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, redirect, url_for
from model import load_and_preprocess_data, model_prediction
import librosa

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Route for Home Page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/teaminfo')
def teaminfo():
    return render_template('teaminfo.html')

# Route for Index Page (file upload)
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        
        if file.filename == '':
            return "No file selected", 400
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Redirect to prediction route
        return redirect(url_for('predict', filename=file.filename))
    
    return render_template('index.html')

# Route for Prediction
@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Preprocess and predict
    X_test = load_and_preprocess_data(file_path)
    result_index = model_prediction(X_test)
    
    # Genre labels
    labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Display the result
    genre = labels[result_index]
    
    return render_template('result.html', genre=genre, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
