# fastapi_stream_wav.py
import os
import io
import time
import wave
import asyncio
import threading
import traceback
import timeit
import base64
from typing import Optional, Any, Literal, List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Body, Query, Form, File, UploadFile
from fastapi.responses import StreamingResponse, ORJSONResponse, Response, RedirectResponse

import torch
import numpy as np
from contextlib import asynccontextmanager

from .config import config, LOGGER_ACCESS, LOGGER
from .tools import base64_audio_to_tensor

from indextts.infer_v2 import IndexTTS2
from indextts.utils.streamer import AsyncAudioStreamer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

class DataQueue:

    def __init__(self, max_batch_size: int, model: IndexTTS2):
        self.active_queue: List[Dict[str, Any]] = []
        self.queue: List[Dict[str, Any]] = []
        self.max_batch_size = max_batch_size
        self.model = model
        self.stopped = False
    
    def put(
        self,
        text: str,
        audio_base64: Optional[str],
        emo_audio_base64: Optional[str],
        emo_vector: Optional[List[float]],
        emo_text: Optional[str],
        lang: str,
        speaker: str,
        temperature: float,
        top_p: float,
        audio_streamer: AsyncAudioStreamer):

        if audio_base64 is not None:
            spk_audio_prompt = base64_audio_to_tensor(
                audio_base64,
                max_length_sec=15.0,
                mono=True,
                target_sr=16000,
                dtype="float32",
            )[0]
        else:
            spk_audio_prompt = voice_list[f"{lang}-{speaker}"].as_posix()
        if emo_audio_base64 is not None:
            emo_audio_prompt = base64_audio_to_tensor(
                emo_audio_base64,
                max_length_sec=15.0,
                mono=True,
                target_sr=16000,
                dtype="float32",
            )[0]
        if emo_vector is not None:
            emo_vector = model.normalize_emo_vec(emo_vector)
        
        item = self.model._init_step_input(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            emo_audio_prompt=emo_audio_prompt if emo_audio_base64 is not None else None,
            emo_vector=emo_vector,
            emo_text=emo_text,
            use_emo_text=emo_text is not None,
            audio_streamer=audio_streamer,
            temperature=temperature,
            top_p=top_p
        )
        if len(self.active_queue) < self.max_batch_size:
            self.active_queue.append(item)
        else:
            self.queue.append(item)
    
    def check_queue(self) -> bool:
        "remove finished items from active queue, fill from waiting queue, but remove cancelled items first from queue"
        removed_count = 0
        for i in range(len(self.active_queue)-1, -1, -1):
            if self.active_queue[i]['finished']:
                self.active_queue.pop(i)
                removed_count += 1
        for i in range(len(self.queue)-1, -1, -1):
            if self.queue[i]['finished']:
                self.queue.pop(i)
        for _ in range(removed_count):
            if self.queue:
                item = self.queue.pop(0)
                self.active_queue.append(item)
        return len(self.active_queue) > 0
    
    @property
    def is_empty(self):
        return len(self.active_queue) == 0
    
    def log_status(self):
        LOGGER.info(f"DataQueue status: active={len(self.active_queue)}, waiting={len(self.queue)}")

    def replace_item(self, index: int, item: Dict[str, Any]):
        if 0 <= index < len(self.active_queue):
            self.active_queue[index] = item

    def set_stopped(self, stopped: bool):
        self.stopped = stopped
    
    def infinite_loop_step(self):
        "infinite loop in another thread and the exit if interrupted or get killed"

        log_counter = 0
        try:
            while True:
                if self.stopped:
                    break
                now = time.time()
                if log_counter >= 15.0:
                    self.log_status()
                    log_counter = 0
                if self.is_empty:
                    time.sleep(0.1)
                    log_counter += (time.time() - now)
                    continue
                self.check_queue()
                for i in range(len(self.active_queue)):
                    item = self.active_queue[i]
                    next_inputs = item['inputs']
                    step_output = self.model._single_step_infer(**next_inputs)
                    self.replace_item(i, step_output)
                torch.cuda.empty_cache()
                log_counter += (time.time() - now)
        except KeyboardInterrupt:
            LOGGER.info("infinite_loop_step interrupted, exiting")
        except Exception as exc:
            LOGGER.error(f"infinite_loop_step exception: {traceback.format_exc()}")

executor = ThreadPoolExecutor(max_workers=4)

# --- model placeholders (load your model in startup) ---
model: Optional[IndexTTS2] = None
data_queue: Optional[DataQueue] = None
lock = threading.Lock()
voice_list = {p.stem.split('_')[0]: p for p in Path(os.path.join(ROOT_DIR, 'sample-voices')).glob('*.wav')}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, data_queue, executor

    model = IndexTTS2(
        model_dir=config.model_path,
        cfg_path=os.path.join(config.model_path, 'config.yaml')
    )
    model.eval()
    data_queue = DataQueue(max_batch_size=config.max_batch_size, model=model)
    loop = asyncio.get_event_loop()
    infinite_thread = loop.run_in_executor(executor, data_queue.infinite_loop_step)
    LOGGER.info("Startup: model should be loaded here")
    yield
    data_queue.set_stopped(True)
    await infinite_thread
    LOGGER.info("Shutdown: clean up resources if needed")

app = FastAPI(title='IndexTTS API',
    description='API for generating text to speech using IndexTTS model with streaming WAV responses.',
    version='1.0.0',
    lifespan=lifespan)


@app.exception_handler(Exception)
async def value_error_handler(request: Request, exc: Exception):
    return ORJSONResponse({
        'error': str(exc),
        'traceback': "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        'status_code': 500
    }, status_code=500)


@app.middleware("http")
async def logging_request(request: Request, call_next):

    client_data = ''
    if request.client:
        client_data = f'{request.client.host}:{request.client.port}'
    LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" START')
    params = str(request.query_params)
    body = await request.body()
    if params:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" PARAMS: {params}')
    if body:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" BODY: {(await request.body())[:256]}')

    start = timeit.default_timer()
    request.state.is_disconnected = request.is_disconnected
    response: Response = await call_next(request)
    response.headers["X-Process-Time"] = f'{timeit.default_timer() - start:.6f}'

    return response


# --- helper to build a complete WAV file from PCM16 bytes ---
def build_wav_from_pcm(pcm_bytes: bytes, sample_rate: int, num_channels: int, sampwidth: int = 2) -> bytes:
    """
    Create a valid WAV file (RIFF) containing pcm_bytes (little-endian PCM16).
    sampwidth is bytes per sample (2 for PCM16).
    """
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    out.seek(0)
    return out.read()


def make_wav_header(
    sample_rate: int,
    num_channels: int,
) -> bytes:
    """
    Build a simple WAV (RIFF) header for PCM (little-endian).
    Note: many clients will accept a header with 0 data_size for chunked streaming.
    """

    num_channels = 1
    sample_width = 2
    frame_rate = sample_rate

    wav_header = io.BytesIO()
    with wave.open(wav_header, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)

    wav_header.seek(0)
    wave_header_bytes = wav_header.read()
    wav_header.close()

    # Create a new BytesIO with the correct MIME type for Firefox
    final_wave_header = io.BytesIO()
    final_wave_header.write(wave_header_bytes)
    final_wave_header.seek(0)

    return final_wave_header.getvalue()



# converter: tensor/array -> PCM16 bytes (mono/stereo)
def chunk_to_pcm16_bytes(chunk: Any, num_channels: int) -> bytes:
    """
    Accepts:
        - torch.Tensor (1D or 2D)
        - numpy array
        - bytes (raw)
    Returns little-endian PCM16 bytes.
    """
    if isinstance(chunk, torch.Tensor):
        arr = chunk.detach().cpu().float().numpy()
    elif isinstance(chunk, np.ndarray):
        arr = chunk
    elif isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    else:
        # fallback: string repr
        return str(chunk).encode("utf-8")

    # If chunk is multi-dimensional (e.g., [frames, channels]), handle channels
    if arr.ndim == 1:
        # mono
        samples = arr
    elif arr.ndim == 2:
        # shape (frames, channels) or (channels, frames) — try to detect common case
        if arr.shape[1] == num_channels:
            # e.g., (frames, channels)
            samples = arr
        elif arr.shape[0] == num_channels:
            # e.g., (channels, frames) -> transpose
            samples = arr.T
        else:
            # unknown layout -> flatten
            samples = arr.flatten()
    else:
        samples = arr.flatten()

    # Normalize & convert floats to int16
    if np.issubdtype(samples.dtype, np.floating):
        # clamp to -1..1 then scale
        clipped = np.clip(samples, -1.0, 1.0)
        int16 = (clipped * 32767).astype(np.int16)
    else:
        # integer types: convert/rescale if needed — here we cast to int16 directly
        int16 = samples.astype(np.int16)

    return int16.tobytes()


@app.get("/tts")
async def gen_wav(
    request: Request,
    text: str = Query(...),
    speaker: Literal["alloy", "ash", "echo", "nova"] = Query("alloy"),
    lang: Literal["en", "id"] = Query('id'),
    prompt: Optional[str] = Query(None),
    temperature: float = Query(1.0, ge=0.0, le=2.0),
    top_p: float = Query(1.0, le=1.0, ge=0.0),
    do_stream: bool = Query(False)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, None, None, prompt, speaker, lang, temperature, top_p, do_stream)

@app.post("/tts")
async def generate_wav(
    request: Request,
    text: str = Body(...),
    speaker: Literal["alloy", "ash", "echo", "nova"] = Body("alloy"),
    lang: Literal["en", "id"] = Body('id'),
    emo_vector: Optional[List[float]] = Body(None),
    prompt: Optional[str] = Body(None),
    temperature: float = Body(1.0, ge=0.0, le=2.0),
    top_p: float = Body(1.0, le=1.0, ge=0.0),
    do_stream: bool = Body(False)
):
    """
    Streams a TTS response produced by the model.
    - only one caller at a time (asyncio.Lock).
    - cooperative stop via audio_streamer.end().
    - streaming binary WAV via chunked transfer.
    """

    return await tts_streamer(request, text, None, emo_vector, prompt, speaker, lang, temperature, top_p, do_stream)

@app.post("/voice_clone")
async def voice_clone_form(
    request: Request,
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    emo_vector: Optional[List[float]] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: float = Form(1.0),
    top_p: float = Form(1.0),
    do_stream: bool = Form(False)
):
    """
    Clones a voice from the provided audio file in form upload.
    """
    audio_bytes = await audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return await tts_streamer(request, text, audio_base64, emo_vector, prompt, "alloy", "id", temperature, top_p, do_stream)

async def tts_streamer(
    request: Request,
    text: str,
    audio_base64: Optional[str],
    emo_vector: Optional[List[float]],
    emo_text: Optional[str],
    speaker: Literal["alloy", "ash", "echo", "nova"],
    lang: Literal["en", "id"],
    temperature: float,
    top_p: float,
    do_stream: bool
):
    global data_queue

    batch_size = 1  # streaming a single sample per request; adapt for batching
    sample_rate = 24000
    num_channels = 1

    # instantiate streamer (adjust signature if needed)
    audio_streamer = AsyncAudioStreamer(batch_size=batch_size, timeout=1.0)  # type: ignore

    data_queue.put(text, audio_base64, None, emo_vector, emo_text, lang, speaker, temperature, top_p, audio_streamer)

    async def disconnect_watcher():
        try:
            is_disc = await request.state.is_disconnected()
        except Exception:
            # treat errors as disconnected
            is_disc = True
        if is_disc:
            audio_streamer.end()

    disconnect_task = asyncio.create_task(disconnect_watcher())

    if not do_stream:
        pcm_chunks = []
        try:
            async for batch_chunks in audio_streamer:
                if 0 in batch_chunks:
                    chunk = batch_chunks[0]
                    pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                    if pcm_bytes:
                        pcm_chunks.append(pcm_bytes)
                else:
                    for idx in sorted(batch_chunks.keys()):
                        chunk = batch_chunks[idx]
                        pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                        if pcm_bytes:
                            pcm_chunks.append(pcm_bytes)
            # finished streaming; assemble all pcm bytes
            all_pcm = b"".join(pcm_chunks)
            full_wav = build_wav_from_pcm(all_pcm, sample_rate, num_channels, sampwidth=2)
            headers = {
                "Content-Type": "audio/wav",
                "Content-Length": str(len(full_wav))
            }
            return Response(content=full_wav, media_type="audio/wav", headers=headers)
        except asyncio.CancelledError:
            audio_streamer.end()
            raise
        except Exception:
            audio_streamer.end()
            raise
        finally:
            audio_streamer.end()

    async def stream_generator():
        """
        Yields:
            - initial WAV header bytes (with placeholder sizes)
            - then PCM16 chunk bytes as received from audio_streamer
            - finally the generate() result metadata as a small JSON/text chunk (optional)
        """
        try:
            # send WAV header first
            header = make_wav_header(sample_rate, num_channels)
            yield header

            # iterate over audio_streamer async iterator (your AsyncAudioStreamer.__aiter__)
            async for batch_chunks in audio_streamer:
                # batch_chunks: dict mapping sample_idx -> chunk
                # We assume batch_size==1 for simplicity; stream only index 0 if present
                # But handle multiple channels if your chunk data contains them.
                if 0 in batch_chunks:
                    chunk = batch_chunks[0]
                    pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                    if pcm_bytes:
                        yield pcm_bytes
                else:
                    # if other indices present, process in increasing order and interleave their PCM
                    for idx in sorted(batch_chunks.keys()):
                        chunk = batch_chunks[idx]
                        pcm_bytes = chunk_to_pcm16_bytes(chunk, num_channels)
                        if pcm_bytes:
                            yield pcm_bytes

                # let event loop schedule (helps responsiveness)
                # await asyncio.sleep(0)

            # optionally yield small trailing info as text chunk (not part of wav)
            # Many clients will ignore data after WAV; if you want strictly valid WAV only, omit this.
            # We omit extra trailing bytes to keep stream pure WAV.
            return  # end generator; connection closes naturally
        except asyncio.CancelledError:
            # generator cancelled (client disconnected); ensure stop flag
            audio_streamer.end()
            raise
        except Exception as exc:
            # On error, set stop flag and re-raise so client sees connection drop
            audio_streamer.end()
            raise
        finally:
            audio_streamer.end()

    # Build binary streaming response with WAV mime type
    return StreamingResponse(stream_generator(), media_type="audio/wav")

@app.get('/', include_in_schema=False)
async def redirect():

    # return ORJSONResponse({'title': app.title, 'description': app.description, 'version': app.version})
    return RedirectResponse(app.root_path+'/docs')
