from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import io
import soundfile as sf
import pandas as pd

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


@app.post("/")
async def analyze_audio(req: AudioRequest):

    audio_bytes = base64.b64decode(req.audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)

    y, sr = sf.read(audio_buffer)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    df = pd.DataFrame({"amplitude": y})

    stats = {
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "variance": df.var().to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict(),
        "median": df.median().to_dict(),
        "mode": df.mode().iloc[0].to_dict(),
        "range": (df.max() - df.min()).to_dict(),
        "allowed_values": {},
        "value_range": {"amplitude": [float(df.min()[0]), float(df.max()[0])]},
        "correlation": []
    }

    return stats