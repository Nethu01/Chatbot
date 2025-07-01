from flask import Flask, request, jsonify
import torch
import random
import json
import os

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Initialize Flask app
app = Flask(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents JSON
INTENTS_PATH = os.getenv("INTENTS_PATH", "intents.json")
with open(INTENTS_PATH, 'r') as json_data:
    intents = json.load(json_data)

# Load trained model data
MODEL_PATH = os.getenv("MODEL_PATH", "data.pth")
data = torch.load(MODEL_PATH)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I do not understand..."

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'message' not in data:
        return jsonify({'error': 'Missing message parameter'}), 400

    message = data['message']
    response = get_response(message)
    return jsonify({'bot_name': bot_name, 'response': response})

if __name__ == "__main__":
    # Use environment variables for configuration
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        debug=os.getenv("FLASK_ENV") == "development"
    )