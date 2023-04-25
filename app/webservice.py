from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import whisper
from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON
from whisper import tokenizer
from faster_whisper import WhisperModel
from .fw.utils import (
    model_converter as faster_whisper_model_converter,
    ResultWriter as faster_whisper_ResultWriter,
    WriteTXT as faster_whisper_WriteTXT,
    WriteSRT as faster_whisper_WriteSRT,
    WriteVTT as faster_whisper_WriteVTT,
    WriteTSV as faster_whisper_WriteTSV,
    WriteJSON as faster_whisper_WriteJSON,
)
import os
from os import path
import ffmpeg
from typing import BinaryIO, Union
import numpy as np
from io import StringIO
from threading import Lock
import torch
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# import importlib.metadata

SAMPLE_RATE = 16000
LANGUAGE_CODES = sorted(list(tokenizer.LANGUAGES.keys()))

CACHE_PATH = os.environ.get('CACHE_PATH', '/root/.cache/')
MODEL = os.environ.get("ASR_MODEL", "large-v2")

# projectMetadata = importlib.metadata.metadata('whisper-asr-webservice')
app = FastAPI(
    # title=projectMetadata['Name'].title().replace('-', ' '),
    # description=projectMetadata['Summary'],
    # version=projectMetadata['Version'],
    # contact={
    #     "url": projectMetadata['Home-page']
    # },
    # license_info={
    #     "name": "MIT License",
    #     "url": projectMetadata['License']
    # }
    title="Whisper ASR Webservice",
    description="A webservice for Whisper ASR",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")


    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )


    applications.get_swagger_ui_html = swagger_monkey_patch

faster_whisper_model_path = os.path.join(CACHE_PATH, "faster_whisper", MODEL)
print(faster_whisper_model_path)
faster_whisper_model_converter(MODEL, faster_whisper_model_path)

if torch.cuda.is_available():
    faster_whisper_model = WhisperModel(MODEL, device="cuda", compute_type="float16")
else:
    faster_whisper_model = WhisperModel(MODEL)
model_lock = Lock()


def get_model():
    return faster_whisper_model


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
def transcribe(
        task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
        language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
        initial_prompt: Union[str, None] = Query(default=None),
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
        output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"])
):
    result = run_asr(audio_file.file, task, language, initial_prompt, encode)
    filename = audio_file.filename.split('.')[0]
    myFile = StringIO()
    write_result(result, myFile, output)
    myFile.seek(0)
    return StreamingResponse(myFile, media_type="text/plain",
                             headers={'Content-Disposition': f'attachment; filename="{filename}.{output}"'})


@app.post("/detect-language", tags=["Endpoints"])
def language_detection(
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg")
):
    # load audio and pad/trim it to fit 30 seconds
    audio = load_audio(audio_file.file, encode)
    audio = whisper.pad_or_trim(audio)

    # detect the spoken language
    with model_lock:
        model = get_model()
        segments, info = model.transcribe(audio, beam_size=5)
        detected_lang_code = info.language

        result = {"detected_language": tokenizer.LANGUAGES[detected_lang_code], "language_code": detected_lang_code}

    return result


def run_asr(
        file: BinaryIO,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        encode=True
):
    audio = load_audio(file, encode)
    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    with model_lock:
        model = get_model()
        segments = []
        text = ""
        i = 0
        segment_generator, info = model.transcribe(audio, beam_size=5, **options_dict)
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        result = {
            "language": options_dict.get("language", info.language),
            "segments": segments,
            "text": text,
        }

    return result


def write_result(
        result: dict, file: BinaryIO, output: Union[str, None]
):
    if (output == "srt"):
        faster_whisper_WriteSRT(ResultWriter).write_result(result, file=file)
    elif (output == "vtt"):
        faster_whisper_WriteVTT(ResultWriter).write_result(result, file=file)
    elif (output == "tsv"):
        faster_whisper_WriteTSV(ResultWriter).write_result(result, file=file)
    elif (output == "json"):
        faster_whisper_WriteJSON(ResultWriter).write_result(result, file=file)
    elif (output == "txt"):
        faster_whisper_WriteTXT(ResultWriter).write_result(result, file=file)
    else:
        return 'Please select an output format!'


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
