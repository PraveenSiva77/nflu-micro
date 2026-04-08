import os
import logging
import math
import shutil
import subprocess
import tempfile
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

# ---------------------------------------------------------------------------
# Configuration from environment variables (with sensible defaults for CPU)
# ---------------------------------------------------------------------------
MODEL_SIZE    = os.getenv("WHISPER_MODEL", "small")        # tiny | base | small | medium | large-v3
DEVICE        = os.getenv("WHISPER_DEVICE", "cpu")         # cpu | cuda
COMPUTE_TYPE  = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 | float16 | float32
CPU_THREADS   = int(os.getenv("WHISPER_CPU_THREADS", "4"))
BEAM_SIZE     = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
MAX_UPLOAD_MB = int(os.getenv("WHISPER_MAX_UPLOAD_MB", "500"))

# ---------------------------------------------------------------------------
# Optional pure-Python audio decoders (fallback when PyAV fails)
# ---------------------------------------------------------------------------
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from scipy.io import wavfile as scipy_wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _resample_linear(data: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    """Simple linear interpolation resample to dst_sr."""
    if src_sr == dst_sr:
        return data
    new_len = math.ceil(len(data) * dst_sr / src_sr)
    return np.interp(
        np.linspace(0, len(data) - 1, new_len),
        np.arange(len(data)),
        data,
    ).astype(np.float32)


def _decode_audio_numpy(path: str) -> Optional[np.ndarray]:
    """
    Try to read *path* as a WAV file using soundfile, scipy, or the stdlib
    wave module and return a float32 numpy array at 16 kHz mono.
    Returns None if all methods fail.
    """
    # --- soundfile ---
    if SOUNDFILE_AVAILABLE:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            data = data.mean(axis=1)
            data = _resample_linear(data, sr)
            logger.info("Decoded audio via soundfile (sr=%d → 16000, mono)", sr)
            return data
        except Exception as e:
            logger.warning("soundfile failed: %s", e)

    # --- scipy ---
    if SCIPY_AVAILABLE:
        try:
            sr, data = scipy_wavfile.read(path)
            if data.dtype != np.float32:
                max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
                data = data.astype(np.float32) / max_val
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = _resample_linear(data, sr)
            logger.info("Decoded audio via scipy (sr=%d → 16000, mono)", sr)
            return data
        except Exception as e:
            logger.warning("scipy failed: %s", e)

    # --- stdlib wave (PCM only, always available) ---
    try:
        import wave, struct
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        # Unpack to int array
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if sampwidth not in dtype_map:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        data = np.frombuffer(raw, dtype=dtype_map[sampwidth]).astype(np.float32)
        max_val = float(np.iinfo(dtype_map[sampwidth]).max)
        data /= max_val
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        data = _resample_linear(data, sr)
        logger.info("Decoded audio via stdlib wave (sr=%d → 16000, mono)", sr)
        return data
    except Exception as e:
        logger.warning("stdlib wave failed: %s", e)

    return None


def _reencode_with_ffmpeg(src: str) -> str:
    """
    Re-encode *src* to a standard 16 kHz mono PCM WAV using ffmpeg.
    Returns path to a new temp file (caller must delete it).
    """
    if not FFMPEG_AVAILABLE:
        raise RuntimeError(
            "Audio format not supported. "
            "Install ffmpeg to handle non-standard audio formats: https://ffmpeg.org/download.html"
        )
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        try:
            os.unlink(out.name)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg re-encoding failed:\n{result.stderr[-1000:]}")
    logger.info("Re-encoded audio via ffmpeg → %s", out.name)
    return out.name


# ---------------------------------------------------------------------------
# Model lifecycle – load once on startup, shared across all requests
# ---------------------------------------------------------------------------
model: Optional[WhisperModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(
        f"Loading WhisperModel  model={MODEL_SIZE}  device={DEVICE}  compute_type={COMPUTE_TYPE}"
    )
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
    )
    logger.info("Model loaded and ready.")
    fallbacks = []
    if SOUNDFILE_AVAILABLE:
        fallbacks.append("soundfile")
    if SCIPY_AVAILABLE:
        fallbacks.append("scipy")
    if FFMPEG_AVAILABLE:
        fallbacks.append("ffmpeg")
    if fallbacks:
        logger.info("Audio fallback decoders available: %s", ", ".join(fallbacks))
    else:
        logger.warning(
            "No fallback decoders found (soundfile, scipy, ffmpeg). "
            "Only standard mp3/flac/wav formats are guaranteed to work. "
            "Run: pip install soundfile scipy"
        )
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Faster-Whisper Speech-to-Text API",
    description=(
        "Upload an audio file and receive a full transcription with per-segment timestamps. "
        "Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend portal (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Utility"])
async def health():
    """Returns 200 OK if the service is running and the model is loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "ffmpeg": FFMPEG_AVAILABLE,
        "soundfile": SOUNDFILE_AVAILABLE,
        "scipy": SCIPY_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Audio Webhook Sessions State
# ---------------------------------------------------------------------------
class AudioSessionCreate(BaseModel):
    user_id: str

audio_sessions_db = {}

# ---------------------------------------------------------------------------
# Core Transcription Runner (Async Wrapper)
# ---------------------------------------------------------------------------
async def process_audio_file_core(
    content: bytes,
    filename: str,
    language: Optional[str],
    task: str,
    word_timestamps: bool,
    vad_filter: bool,
    beam_size: int
) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Please retry.")

    # Sanitize language
    VALID_LANGUAGES = {
        "af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs",
        "cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu",
        "ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka",
        "kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml",
        "mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt",
        "ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw",
        "ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo",
        "zh","yue",
    }
    lang: Optional[str] = None
    if language:
        lang_clean = language.strip().lower()
        if lang_clean and lang_clean != "string":
            if lang_clean not in VALID_LANGUAGES:
                raise HTTPException(status_code=422, detail=f"Invalid language code '{language}'.")
            lang = lang_clean

    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_MB} MB.")

    suffix = os.path.splitext(filename or "audio")[-1] or ".tmp"
    tmp_path: Optional[str] = None
    reencoded_path_local: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Transcribing internally: {filename}  ({size_mb:.1f} MB)")

        def _execute_transcription():
            nonlocal reencoded_path_local
            try:
                def _run_transcribe(audio_input):
                    seg_gen, inf = model.transcribe(
                        audio_input, language=lang, task=task, beam_size=beam_size,
                        word_timestamps=word_timestamps, vad_filter=vad_filter,
                    )
                    return _collect_segments(seg_gen, word_timestamps), inf
                try:
                    segments, info = _run_transcribe(tmp_path)
                except Exception as decode_err:
                    err_str = str(decode_err).lower()
                    is_decode_err = any(k in err_str for k in ("invalid data", "errno 1094995529", "decode", "no such file"))
                    if not is_decode_err: raise
                    logger.warning("PyAV decode failed – trying fallback.")
                    audio_np = _decode_audio_numpy(tmp_path)
                    if audio_np is not None:
                        segments, info = _run_transcribe(audio_np)
                    else:
                        logger.warning("Pure-Python fallback failed – trying ffmpeg.")
                        reencoded_path_local = _reencode_with_ffmpeg(tmp_path)
                        segments, info = _run_transcribe(reencoded_path_local)

                return {
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "duration": round(info.duration, 3),
                    "segments": segments,
                    "text": " ".join(s["text"] for s in segments),
                }
            except Exception as e:
                logger.error("Core transcription error: %s", e)
                raise e

        # Execute off event loop
        payload = await asyncio.to_thread(_execute_transcription)
        return payload

    finally:
        for path in (tmp_path, reencoded_path_local):
            if path and os.path.exists(path):
                try: os.unlink(path)
                except: pass

# ---------------------------------------------------------------------------
# Endpoints: Webhook Listener Sessions
# ---------------------------------------------------------------------------
@app.post("/api/audio-sessions", tags=["Webhook Listener"])
async def create_audio_session(session_data: AudioSessionCreate):
    """Generates a unique session webhook for external audio payload processing."""
    session_id = str(uuid.uuid4())
    audio_sessions_db[session_id] = {
        "user_id": session_data.user_id,
        "status": "PENDING",
        "data": None
    }
    return {
        "session_id": session_id,
        "webhook_url": f"/webhook/{session_id}"
    }

@app.get("/api/audio-sessions/{session_id}/status", tags=["Webhook Listener"])
async def get_audio_session_status(session_id: str):
    """Retrieve the current processing status of an active audio webhook session."""
    session = audio_sessions_db.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session

@app.post("/webhook/{session_id}", tags=["Webhook Listener"])
async def handle_webhook_payload(
    session_id: str,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    word_timestamps: bool = Form(False),
    vad_filter: bool = Form(True),
    beam_size: int = Form(BEAM_SIZE),
):
    """Receives an audio payload from an external service, processes it, and updates internal session state."""
    session = audio_sessions_db.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid webhook session ID.")
    
    session["status"] = "PROCESSING"
    try:
        content = await file.read()
        result = await process_audio_file_core(
            content=content,
            filename=file.filename or session_id,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            beam_size=beam_size
        )
        session["status"] = "COMPLETED"
        session["data"] = result
        return result
    except Exception as e:
        session["status"] = "ERROR"
        session["data"] = {"error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Transcription endpoint
# ---------------------------------------------------------------------------
@app.post("/transcribe", tags=["Transcription"])
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to transcribe (mp3, wav, flac, m4a, ogg, etc.)"),
    language: Optional[str] = Form(None, description="Language code e.g. 'en', 'fr'. Auto-detected if omitted."),
    task: str = Form("transcribe", description="'transcribe' or 'translate' (to English)."),
    word_timestamps: bool = Form(False, description="Include per-word timestamps."),
    vad_filter: bool = Form(True, description="Use VAD to skip silent regions."),
    beam_size: int = Form(BEAM_SIZE, description="Beam size for decoding."),
    webhook_url: Optional[str] = Form(None, description="URL to receive webhook POST on completion."),
    job_id: Optional[str] = Form(None, description="Client-provided job ID (or generated)."),
):
    """
    Transcribe an uploaded audio file.
    """
    content = await file.read()
    filename = file.filename or "audio"

    async def _external_webhook_worker(target_url: str, j_id: str, file_bytes: bytes, fname: str):
        try:
            payload = await process_audio_file_core(
                file_bytes, fname, language, task, word_timestamps, vad_filter, beam_size
            )
            payload["job_id"] = j_id
            logger.info(f"Transcription complete for job {j_id}, dispatching pushed webhook...")
            async with httpx.AsyncClient() as client:
                await client.post(target_url, json=payload, timeout=60.0)
            logger.info(f"Deployed payload to {target_url} successfully.")
        except Exception as e:
            logger.error(f"Failed to push webhook payload for job {j_id}: {e}")

    if webhook_url:
        current_job_id = job_id or str(uuid.uuid4())
        background_tasks.add_task(_external_webhook_worker, webhook_url, current_job_id, content, filename)
        return JSONResponse(status_code=202, content={"status": "processing", "job_id": current_job_id})

    # Sync flow
    try:
        payload = await process_audio_file_core(
            content, filename, language, task, word_timestamps, vad_filter, beam_size
        )
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _collect_segments(segments_gen, word_timestamps: bool) -> list:
    """Materialise the lazy faster-whisper segment generator into a plain list."""
    segments = []
    for seg in segments_gen:
        seg_data = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        if word_timestamps and seg.words:
            seg_data["words"] = [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                for w in seg.words
            ]
        segments.append(seg_data)
    return segments
