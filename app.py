from fastapi import FastAPI, Request
import pickle
import torch

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()  # Set model to evaluation mode

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hugging Face model deployed on Render!"}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        input_data = data.get("input")
        if input_data is None:
            return {"error": "Missing 'input' in request"}
        
        input_tensor = torch.tensor(input_data)  # ensure it's the right shape
        with torch.no_grad():
            output = model(input_tensor)
        
        return {"output": output.tolist()}
    except Exception as e:
        return {"error": str(e)}
