from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

LATEST = {}

class Telemetry(BaseModel):
    timestamp: str
    safe: float
    rocks: float
    crater: float
    source: str

@app.post("/push")
def push_data(data: Telemetry):
    global LATEST
    LATEST = data.dict()
    return {"status": "ok"}

@app.get("/latest")
def latest():
    if LATEST:
        return LATEST
    return {"status": "no data"}
