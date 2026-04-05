from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import io
import soundfile as sf
from collections import Counter

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def compute_mode(arr):
    counts = Counter(arr)
    return counts.most_common(1)[0][0]


@app.post("/")
async def analyze_audio(req: AudioRequest):

    # decode base64
    audio_bytes = base64.b64decode(req.audio_base64)

    # load audio
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = sf.read(audio_buffer)

    # convert stereo to mono if needed
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {
        "mfcc_mean": np.mean(mfcc),
        "mfcc_std": np.std(mfcc),
        "mfcc_var": np.var(mfcc),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y))
    }

    values = np.array(list(features.values()))

    result = {
        "rows": int(values.shape[0]),
        "columns": list(features.keys()),

        "mean": {k: float(np.mean([v])) for k,v in features.items()},
        "std": {k: float(np.std([v])) for k,v in features.items()},
        "variance": {k: float(np.var([v])) for k,v in features.items()},
        "min": {k: float(v) for k,v in features.items()},
        "max": {k: float(v) for k,v in features.items()},
        "median": {k: float(np.median([v])) for k,v in features.items()},
        "mode": {k: float(v) for k,v in features.items()},
        "range": {k: 0.0 for k in features.keys()},

        "allowed_values": {k: [] for k in features.keys()},
        "value_range": {k: [float(v), float(v)] for k,v in features.items()},

        "correlation": []
    }

    return result