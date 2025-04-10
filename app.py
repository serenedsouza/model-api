from fastapi import FastAPI, Request
import pickle
import torch

# Load model
with open("huggingface_model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()  # Set model to evaluation mode

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hugging Face model deployed on Render!"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    input_tensor = torch.tensor(data["input"])  # assuming proper formatting
    with torch.no_grad():
        output = model(input_tensor)
    return {"output": output.tolist()}
