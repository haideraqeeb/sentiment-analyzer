from fastapi import FastAPI, Request
import transformers
import uvicorn
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

app = FastAPI()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    if "text" in data:
        user_input = data["text"]
        inputs = tokenizer([user_input], padding="max_length", truncation=True, return_tensors="pt", max_length=512)
        output=model(**inputs)
        y_pred = np.argmax(output.logits.detach().numpy(),axis=1)  
        response = {"Recieved Text": user_input,"Prediction": y_pred}
    else:
        response = {"Recieved Text": "No text recieved."}
    return response


if __name__ == "__main__":
    uvicorn.run("main:app",host='0.0.0.0', port=8080, reload=True, debug=True)