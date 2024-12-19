from flask import Flask, request, jsonify, render_template
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model_path = r'C:\PyCharmProjects\fakejobopostings\fakejobopostings\model\jobposting_classification_model.h5'
model = load_model(model_path)

# Load tokenizer settings
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 502
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

# Predefined tokenizer vocabulary (optional: replace with the one used during training)
# For accurate predictions, save and load the exact tokenizer used in training.
# tokenizer.fit_on_texts(saved_vocabulary)

def preprocess_text(input_text):
    """
    Function to preprocess input text to match the model's requirements.
    """
    # Remove non-alphabetic characters and convert to lowercase
    pattern = "[^a-zA-Z]"
    cleaned_text = re.sub(pattern, " ", input_text).lower()

    # Convert text to sequences and pad
    tokenized = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(tokenized, maxlen=MAX_SEQUENCE_LENGTH)
    return padded

@app.route('/')
def home():
    """
    Render the homepage with an input form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle text input, process it, and return prediction.
    """
    # Get input from form
    input_text = request.form['input_text']

    # Preprocess the input
    preprocessed_input = preprocess_text(input_text)

    # Predict using the model
    prediction = model.predict(preprocessed_input)
    result = "Fraudulent" if prediction[0][0] > 0.5 else "Non-Fraudulent"

    return jsonify({
        'Input Text': input_text,
        'Prediction': result,
        'Confidence Score': float(prediction[0][0])
    })

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
