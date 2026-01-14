from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
latest_data = {}

class Payload(BaseModel):
    timestamp: str
    safe: float
    rocks: float
    crater: float
    source: str

@app.post("/update")
def update(data: Payload):
    global latest_data
    latest_data = data.dict()
    return {"status": "received"}

@app.get("/latest")
def latest():
    return latest_data
