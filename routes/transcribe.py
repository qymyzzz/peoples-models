import io

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

router = APIRouter()


pipe = pipeline(
    model="qymyz/whisper-tiny-russian-dysarthria",
)


class TranscriptionResult(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(file: UploadFile = File(...)):
    audio_data, samplerate = sf.read(io.BytesIO(await file.read()))
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    audio_data = np.array(audio_data, dtype=np.float32)

    result = pipe(audio_data)

    return JSONResponse(content={"text": result["text"]})
