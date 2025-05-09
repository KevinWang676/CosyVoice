# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
#from cosyvoice.utils.file_utils import load_wav

from fastapi import HTTPException
import requests
import tempfile
import torchaudio

# Keep the original load_wav function unchanged
def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

# Add a new function to handle URLs
def load_wav_from_url(url, target_sr):
    # Download the file from the URL to a temporary file
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {url}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file.flush()
        temp_path = temp_file.name
    
    try:
        # Use the existing load_wav function with the file path
        speech, sample_rate = torchaudio.load(temp_path, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(), 
    prompt_text: str = Form(), 
    prompt_wav_url: str = Form(...),  # Using ... makes this parameter required
    speed: float = Form(...)
):
    # Process the URL directly - no need for conditionals
    prompt_speech_16k = load_wav_from_url(prompt_wav_url, 16000)
    
    # Rest of the function remains the same
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=speed)
    
    # Collect all audio data instead of streaming it
    audio_data = bytearray()
    for chunk in generate_data(model_output):
        audio_data.extend(chunk)
        
    # Return complete audio file
    return Response(
        content=bytes(audio_data),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
    )

@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))

@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File(), speed: float = Form(...)):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    
    # Disable streaming by setting stream=False (assuming the function accepts this parameter)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=False, speed=speed)
    
    # Collect all audio data instead of streaming it
    audio_data = bytearray()
    for chunk in generate_data(model_output):
        audio_data.extend(chunk)
    print("instruct模式生成成功！")
    # Return complete audio file
    return Response(
        content=bytes(audio_data),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    #try:
    #    cosyvoice = CosyVoice(args.model_dir)
    #except Exception:
    try:
        #cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=True)
        cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=True)
    except Exception:
        raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=8000)
