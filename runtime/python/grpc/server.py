#!/usr/bin/env python3
# coding: utf‑8
"""
CosyVoice gRPC back‑end – updated to mirror the FastAPI logic
*   loads CosyVoice2 with TRT / FP16 first (falls back to CosyVoice)
*   inference_zero_shot  ➜  adds   stream=False   +   speed
*   inference_instruct   ➜  keeps original “speaker‑ID” path
*   inference_instruct2  ➜  new:  prompt‑audio + speed (no speaker‑ID)
"""

import io, tempfile, requests, soundfile as sf, torchaudio
import os
import sys
from concurrent import futures
import argparse
import logging
import grpc
import numpy as np
import torch

import cosyvoice_pb2
import cosyvoice_pb2_grpc

# ────────────────────────────────────────────────────────────────────────────────
# set‑up
# ────────────────────────────────────────────────────────────────────────────────
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    f"{ROOT_DIR}/../../..",
    f"{ROOT_DIR}/../../../third_party/Matcha-TTS",
])

from cosyvoice.cli.cosyvoice import CosyVoice2          # noqa: E402


# ────────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────────
def _bytes_to_tensor(wav_bytes: bytes) -> torch.Tensor:
    """
    Convert int16 little‑endian PCM bytes → torch.FloatTensor in range [‑1,1]
    """
    speech = torch.from_numpy(
        np.frombuffer(wav_bytes, dtype=np.int16)
    ).unsqueeze(0).float() / (2 ** 15)
    return speech                                                      # [1, T]


def _yield_audio(model_output):
    """
    Generator that converts CosyVoice output → protobuf Response messages.
    """
    for seg in model_output:
        pcm16 = (seg["tts_speech"].numpy() * (2 ** 15)).astype(np.int16)
        resp = cosyvoice_pb2.Response(tts_audio=pcm16.tobytes())
        yield resp

import os, io, tempfile, requests, torch, torchaudio
from urllib.parse import urlparse

def _load_prompt_from_url(url: str, target_sr: int = 16_000) -> torch.Tensor:
    """Download an audio file from ``url`` (wav / mp3 / flac / ogg …),
    convert it to mono, resample to ``target_sr`` if necessary,
    and return a 1×T float‑tensor in the range ‑1…1."""
    
    # ─── 1.  Download ────────────────────────────────────────────────────────────
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=400,
                            detail=f"Failed to download audio from URL: {url}")

    # Infer extension from URL *or* Content‑Type header
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if not ext and 'content-type' in resp.headers:
        mime = resp.headers['content-type'].split(';')[0].strip()
        ext = {
            'audio/mpeg': '.mp3',
            'audio/wav':  '.wav',
            'audio/x-wav': '.wav',
            'audio/flac': '.flac',
            'audio/ogg':  '.ogg',
            'audio/x-m4a': '.m4a',
        }.get(mime, '.audio')            # generic fallback

    with tempfile.NamedTemporaryFile(suffix=ext or '.audio', delete=False) as f:
        f.write(resp.content)
        temp_path = f.name

    # ─── 2.  Decode (torchaudio first, pydub fallback) ──────────────────────────
    try:
        # Let torchaudio pick the right backend automatically
        speech, sample_rate = torchaudio.load(temp_path)
    except Exception:
        # Fallback that works as long as ffmpeg is present
        from pydub import AudioSegment
        import numpy as np

        seg = AudioSegment.from_file(temp_path)       # any ffmpeg‑supported format
        seg = seg.set_channels(1)                     # force mono
        sample_rate = seg.frame_rate
        np_audio = np.array(seg.get_array_of_samples()).astype(np.float32)
        # normalise to −1…1 based on sample width
        np_audio /= float(1 << (8 * seg.sample_width - 1))
        speech = torch.from_numpy(np_audio).unsqueeze(0)

    finally:
        os.unlink(temp_path)

    # ─── 3.  Ensure mono + correct sample‑rate ──────────────────────────────────
    if speech.dim() > 1 and speech.size(0) > 1:
        speech = speech.mean(dim=0, keepdim=True)     # average to mono

    if sample_rate != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                new_freq=target_sr)(speech)
    return speech
        
# ────────────────────────────────────────────────────────────────────────────────
# gRPC service
# ────────────────────────────────────────────────────────────────────────────────
class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        # try CosyVoice2 first (preferred runtime: TRT / FP16)
        try:
            self.cosyvoice = CosyVoice2(args.model_dir,
                                        load_jit=False,
                                        load_trt=True,
                                        fp16=True)
            logging.info("Loaded CosyVoice2 (TRT / FP16).")
        except Exception:
            raise TypeError("No valid CosyVoice model found!")

    # ---------------------------------------------------------------------
    # single bi‑di streaming RPC
    # ---------------------------------------------------------------------
    def Inference(self, request, context):
        """Route to the correct model call based on the oneof field present."""
        # 1. Supervised fine‑tuning
        if request.HasField("sft_request"):
            logging.info("Received SFT inference request")
            mo = self.cosyvoice.inference_sft(
                request.sft_request.tts_text,
                request.sft_request.spk_id
            )
            yield from _yield_audio(mo)
            return

        # 2. Zero‑shot speaker cloning  (bytes OR S3 URL)
        if request.HasField("zero_shot_request"):
            logging.info("Received zero‑shot inference request")
            zr = request.zero_shot_request
            tmp_path = None  # initialise so we can delete later
        
            try:
                # ───── determine payload type ──────────────────────────────────────
                if zr.prompt_audio.startswith(b'http'):
                    prompt = _load_prompt_from_url(zr.prompt_audio.decode('utf‑8'))
                else:
                    # —— legacy raw PCM bytes —— -----------------------------------
                    prompt = _bytes_to_tensor(zr.prompt_audio)
        
                # ───── call the model ──────────────────────────────────────────────
                speed = getattr(zr, "speed", 1.0)
                mo = self.cosyvoice.inference_zero_shot(
                    zr.tts_text,
                    zr.prompt_text,
                    prompt,
                    stream=False,
                    speed=speed,
                )
          
            finally:
                # clean up any temporary file we created
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logging.warning("Could not remove temp file %s: %s", tmp_path, e)

            yield from _yield_audio(mo)
            return
      
        # 3. Cross‑lingual
        if request.HasField("cross_lingual_request"):
            logging.info("Received cross‑lingual inference request")
            cr = request.cross_lingual_request
            tmp_path = None
        
            try:
                if cr.prompt_audio.startswith(b'http'):          # S3 URL case
                    prompt = _load_prompt_from_url(cr.prompt_audio.decode('utf‑8'))        
                else:                                           # legacy raw bytes
                    prompt = _bytes_to_tensor(cr.prompt_audio)
        
                mo = self.cosyvoice.inference_cross_lingual(
                    cr.tts_text,
                    prompt
                )
        
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logging.warning("Could not remove temp file %s: %s",
                                        tmp_path, e)
        
            yield from _yield_audio(mo)
            return


        # 4. Instruct‑2  (CosyVoice2 supports this variant only)
        if request.HasField("instruct_request"):
        
            ir = request.instruct_request
        
            # ---- require that the descriptor contains the field -------------------
            if 'prompt_audio' not in ir.DESCRIPTOR.fields_by_name:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Server expects instruct‑2 proto with a 'prompt_audio' field."
                )
        
            # ---- make sure it is non‑empty (no HasField for proto3 scalars) -------
            if len(ir.prompt_audio) == 0:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "'prompt_audio' must not be empty for instruct‑2 requests."
                )
        
            logging.info("Received instruct‑2 inference request")
        
            # convert to bytes no matter what scalar type the proto uses
            pa_bytes = (ir.prompt_audio.encode('utf-8') if isinstance(ir.prompt_audio, str)
                        else ir.prompt_audio)
        
            # URL vs raw bytes
            if pa_bytes.startswith(b"http"):
                prompt = _load_prompt_from_url(pa_bytes.decode('utf-8'))
            else:
                prompt = _bytes_to_tensor(pa_bytes)
        
            speed = getattr(ir, "speed", 1.0)
            mo = self.cosyvoice.inference_instruct2(
                ir.tts_text,
                ir.instruct_text,
                prompt,
                stream=False,
                speed=speed,
            )
        
            yield from _yield_audio(mo)
            return


        # unknown request type
        context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                      "Unsupported request type in oneof field.")


# ────────────────────────────────────────────────────────────────────────────────
# entry‑point
# ────────────────────────────────────────────────────────────────────────────────
def serve(args):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_conc),
        maximum_concurrent_rpcs=args.max_conc
    )
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(
        CosyVoiceServiceImpl(args), server
    )
    server.add_insecure_port(f"0.0.0.0:{args.port}")
    server.start()
    logging.info("CosyVoice gRPC server listening on 0.0.0.0:%d", args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_conc", type=int, default=4,
                        help="maximum concurrent requests / threads")
    parser.add_argument("--model_dir", type=str,
                        default="pretrained_models/CosyVoice2-0.5B",
                        help="local path or ModelScope repo id")
    serve(parser.parse_args())
