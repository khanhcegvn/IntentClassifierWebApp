from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession("intent_classifier_model.onnx")

class InputData(BaseModel):
    input_values: list[float]

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([data.input_values], dtype=np.float32)
    inputs = {session.get_inputs()[0].name: input_array}
    outputs = session.run(None, inputs)
    return {"prediction": outputs[0].tolist()}
