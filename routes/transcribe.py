import os
import tempfile
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Global variable to hold our pipeline
pipe = None


class TranscriptionResult(BaseModel):
    text: str


def load_model():
    global pipe
    pipe = pipeline(
        model="qymyz/whisper-russian-dysarthria",
        torch_dtype=torch.float16,  # Use half-precision for memory efficiency
        device="cpu",  # Force CPU usage for more consistent performance on Render
    )


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    global pipe

    if pipe is None:
        load_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        result = pipe(temp_file_path)
        background_tasks.add_task(os.unlink, temp_file_path)
        return JSONResponse(content={"text": result["text"]})
    except Exception as e:
        background_tasks.add_task(os.unlink, temp_file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
