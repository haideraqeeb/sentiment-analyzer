from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "climatebert/distilroberta-base-climate-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        text = file.read().decode("utf-8")

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            output = model(**inputs)

        y_pred = np.argmax(output.logits.numpy(), axis=1).tolist()

        response = {"Received Text": text, "Prediction": y_pred}
        return jsonify(response)

    return "Please submit a POST request with a file containing text."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, request
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn

model_name = "climatebert/distilroberta-base-climate-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/", methods=["GET", "POST"])
def index():
    request.method == "POST"
    file = request.files["file"]
    text = file["text"]

    inputs = tokenizer(text)

    with torch.no_grad():
        output = model(**inputs)

    y_pred = np.argmax(output.logits.numpy(), axis=1).tolist()

    response = {"Received Text": text, "Prediction": y_pred}
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

