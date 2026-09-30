import os
import sys
import subprocess
import random
import argparse
import time
import tempfile
import json
import gc
import re
import shutil
import platform
import functools
import traceback
import numpy as np
import glob
from datetime import datetime

import psutil
DEFAULT_CLEAR_CACHE = True if psutil.virtual_memory().total < 31 * 1024**3 else False

# Emoji in library messages must not crash a console with a legacy code page or a redirected log
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

if sys.platform == "win32":
    # A browser tab that closes or reloads resets its connection; asyncio's Windows event loop then prints a
    # "ConnectionResetError: [WinError 10054]" traceback although nothing failed. Keep the console clean.
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    def _quiet_call_connection_lost(self, exc):
        try:
            _call_connection_lost(self, exc)
        except ConnectionResetError:
            pass

    _ProactorBasePipeTransport._call_connection_lost = _quiet_call_connection_lost

import torch
import gradio as gr
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm

from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from diffsynth.models import wan_video_dit
from video_utils import reencode_video_to_16fps, clean_temp_videos, check_video_has_audio, add_audio_to_video
from filelock import FileLock
from ui_theme import CSS, HEAD, TOGGLE_SECTIONS_JS, TOGGLE_THEME_JS, app_theme, btn

APP_VERSION = "V73"
# Gradio 6 takes the theme, CSS and head in launch(); Gradio 5 (older installs) in Blocks()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
PAGE_STYLE = {"theme": app_theme(), "css": CSS, "head": HEAD}
APP_TITLE = f"SECourses Wan 2.1 I2V - V2V - T2V Advanced Gradio APP {APP_VERSION}"
TUTORIAL_URL = "https://youtu.be/hnAhveNy-8s"
PATREON_URL = "https://www.patreon.com/posts/123105403"

DEFAULT_OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
LORAS_DIR = "LoRAs"

MODEL_T2V_1_3B = "WAN 2.1 1.3B (Text/Video-to-Video)"
MODEL_T2V_14B = "WAN 2.1 14B Text-to-Video"
MODEL_I2V_720P = "WAN 2.1 14B Image-to-Video 720P"
MODEL_I2V_480P = "WAN 2.1 14B Image-to-Video 480P"
MODEL_CHOICES = [MODEL_T2V_1_3B, MODEL_T2V_14B, MODEL_I2V_720P, MODEL_I2V_480P]

VRAM_PRESETS = ["4GB", "6GB", "8GB", "10GB", "12GB", "16GB", "24GB", "32GB", "48GB", "80GB"]
DTYPE_CHOICES = ["torch.float8_e4m3fn", "torch.bfloat16"]
RIFE_CHOICES = ["2x FPS", "4x FPS"]
TARGET_LANGUAGES = ["CH", "EN"]

# Attention kernel for the DiT. Sage Attention is about 2.5x faster than Flash Attention on RTX 40/50 GPUs
# (it also computes RoPE in FP32); Flash Attention with FP64 RoPE reproduces V72 results exactly.
ATTENTION_SAGE = "Sage Attention (fastest)"
ATTENTION_FLASH = "Flash Attention (same results as V72)"
ATTENTION_SDPA = "PyTorch SDPA (slowest, most compatible)"
ATTENTION_CHOICES = {
    ATTENTION_SAGE: ("sage", "fp32"),
    ATTENTION_FLASH: ("flash", "fp64"),
    ATTENTION_SDPA: ("sdpa", "fp64"),
}
ATTENTION_NAMES = {"sage": "Sage Attention", "flash": "Flash Attention", "sdpa": "PyTorch SDPA"}

# Model files, relative to models/ (the downloader keeps the shared files once in models/)
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
VAE_FILE = "Wan2.1_VAE.pth"
CLIP_FILE = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
MODEL_FILES = {
    "1.3B": ("Wan2.1-T2V-1.3B", ["diffusion_pytorch_model.safetensors"]),
    "14B_text": ("Wan2.1-T2V-14B", [f"diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)]),
    "14B_image_720p": ("Wan2.1-I2V-14B-720P", [f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)]),
    "14B_image_480p": ("Wan2.1-I2V-14B-480P", [f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)]),
}
MODEL_DOWNLOAD_OPTION = {"1.3B": 1, "14B_image_720p": 2, "14B_image_480p": 3, "14B_text": 4}
QWEN_LOCAL_DIR = os.path.join(MODELS_DIR, "Qwen2.5-14B-Instruct")

# Add cleanup of stale temporary reservation files
def cleanup_temp_reservation_files():
    """Clean up any stale temporary reservation files that might have been left by crashed instances"""
    try:
        temp_dir = DEFAULT_OUTPUT_DIR
        if os.path.exists(temp_dir):
            for tmp_file in glob.glob(os.path.join(temp_dir, "*.tmp")):
                # Check if the file is older than 1 hour - it's likely stale
                file_age = time.time() - os.path.getmtime(tmp_file)
                if file_age > 3600:  # 1 hour in seconds
                    try:
                        os.remove(tmp_file)
                        print(f"[CMD] Removed stale temporary file: {tmp_file}")
                    except Exception as e:
                        print(f"[CMD] Failed to remove stale temporary file {tmp_file}: {e}")
    except Exception as e:
        print(f"[CMD] Error during cleanup of temporary reservation files: {e}")

# Wrapper for save_video that handles temporary reservation files
def safe_save_video(video_data, filename, fps=16, quality=90):
    """
    Wrapper for save_video that ensures proper cleanup of temporary reservation files.
    Also provides atomic file operations to prevent race conditions between instances.

    The filename can be either a string (legacy mode) or a tuple of (filename, temp_filename)
    from the get_next_filename function.
    """
    try:
        # Handle both string and tuple inputs for backward compatibility
        actual_filename = filename
        temp_file = None

        if isinstance(filename, tuple):
            actual_filename, temp_file = filename
        else:
            # Legacy mode - assume the temp file follows the .tmp convention
            temp_file = filename + ".tmp"

        # Save the video
        save_video(video_data, actual_filename, fps=fps, quality=quality)

        # Clean up the temporary reservation file if it exists
        remove_temp_file(temp_file)

        return True, actual_filename
    except Exception as e:
        print(f"[CMD] Error saving video {actual_filename if 'actual_filename' in locals() else filename}: {e}")
        # Clean up temp file on error as well
        if 'temp_file' in locals() and temp_file:
            remove_temp_file(temp_file)
        return False, None

# Call cleanup on startup
cleanup_temp_reservation_files()

# Global variables
loaded_pipeline = None
loaded_pipeline_config = {}
cancel_flag = False
cancel_batch_flag = False
prompt_expander = None
last_selected_aspect_ratio = None
args = None

# ------------------------- Utility Functions -------------------------

def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = None
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if not ret:
            frame = None
    if frame is None:
        # Some encodings cannot seek to the last frame: read through the video instead
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, current = cap.read()
            if not ret:
                break
            frame = current
    cap.release()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None

def generate_prompt_info(parameters):
    """
    Generate prompt info text from parameters dict.
    """
    details = ""
    details += f"Prompt: {parameters['prompt']}\n"
    details += f"Negative Prompt: {parameters['negative_prompt']}\n"

    # Add clarity for extension models
    if 'extension_segment' in parameters and parameters['extension_segment'] > 0:
        details += f"Used Model: {parameters['model_choice']} (Extension Model)\n"
    else:
        details += f"Used Model: {parameters['model_choice']}\n"

    if 'extension_model' in parameters:
        details += f"Extension Model: {parameters['extension_model']}\n"

    details += f"Number of Inference Steps: {parameters['inference_steps']}\n"
    details += f"CFG Scale: {parameters['cfg_scale']}\n"
    details += f"Sigma Shift: {parameters['sigma_shift']}\n"
    details += f"Seed: {parameters['seed']}\n"
    details += f"Number of Frames: {parameters['num_frames']}\n"

    if 'extend_factor' in parameters:
        details += f"Extend Factor: {parameters['extend_factor']}x\n"

    if 'num_segments' in parameters:
        details += f"Number of Segments: {parameters['num_segments']}\n"

    if 'extension_segment' in parameters:
        details += f"Extension Segment: {parameters['extension_segment']} of {parameters['total_extensions']}\n"

    if 'source_frame' in parameters:
        details += f"Used Last Frame From: {parameters['source_frame']}\n"

    if 'input_file' in parameters:
        if parameters.get('is_video', False):
            details += f"Input Video: {parameters['input_file']}\n"
        else:
            details += f"Input Image: {parameters['input_file']}\n"

    if 'denoising_strength' in parameters:
        if parameters.get('is_text_to_video', False) and not parameters.get('has_input_video', False):
            details += "Denoising Strength: N/A\n"
        else:
            details += f"Denoising Strength: {parameters['denoising_strength']}\n"

    if 'pr_rife_enabled' in parameters and parameters['pr_rife_enabled']:
        details += f"Practical-RIFE: Enabled, Multiplier: {parameters.get('pr_rife_multiplier', '(unspecified)')}\n"

    if 'segment_details' in parameters:
        if isinstance(parameters['segment_details'], list):
            for detail in parameters['segment_details']:
                if isinstance(detail, tuple):
                    i, text = detail
                    details += f"Extension segment {i}: {text}\n"
                else:
                    details += f"{detail}\n"

    if 'lora_details' in parameters:
        if parameters['lora_details']:
            details += f"LoRA Models: {parameters['lora_details']}\n"
        else:
            details += "LoRA Model: None\n"

    details += f"TeaCache Enabled: {parameters['enable_teacache']}\n"
    if parameters['enable_teacache']:
        details += f"TeaCache L1 Threshold: {parameters['tea_cache_l1_thresh']}\n"
        details += f"TeaCache Model ID: {parameters['tea_cache_model_id']}\n"

    details += f"Precision: {'FP8' if parameters['torch_dtype'] == 'torch.float8_e4m3fn' else 'BF16'}\n"
    if parameters.get('attention'):
        details += f"Attention: {parameters['attention']}\n"
    details += f"Auto Crop: {'Enabled' if parameters.get('auto_crop', False) else 'Disabled'}\n"
    details += f"Final Resolution: {parameters['width']}x{parameters['height']}\n"

    if 'video_generation_duration' in parameters:
        details += f"Video Generation Duration: {parameters['video_generation_duration']:.2f} seconds"
        if parameters.get('include_minutes', False):
            details += f" / {parameters['video_generation_duration']/60:.2f} minutes"
        details += "\n"

    if 'generation_duration' in parameters:
        details += f"Total Processing Duration: {parameters['generation_duration']:.2f} seconds"
        if parameters.get('include_minutes', False):
            details += f" / {parameters['generation_duration']/60:.2f} minutes"
        details += "\n"

    return details

def remove_temp_file(temp_file):
    """Safely remove a temporary file if it exists"""
    if temp_file and os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print(f"[CMD] Removed temporary file: {temp_file}")
        except Exception as e:
            print(f"[CMD] Failed to remove temporary file {temp_file}: {e}")

# Modify the merge_videos function to use the remove_temp_file function
def merge_videos(video_files, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Merge multiple video files into one, preserving audio if present.
    """
    if not video_files:
        print("[CMD] No video files provided for merging")
        return None

    # Check if any input videos have audio
    has_audio = any(check_video_has_audio(vf) for vf in video_files if os.path.exists(vf))
    print(f"[CMD] Detected audio in input videos: {has_audio}")

    # Create a temporary file list for ffmpeg
    filelist_path = os.path.join(tempfile.gettempdir(), "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as f:
        for vf in video_files:
            if os.path.exists(vf):
                f.write(f"file '{os.path.abspath(vf)}'\n")
            else:
                print(f"[CMD] Warning: file not found for merging: {vf}")

    # Check if filelist is empty
    if os.path.getsize(filelist_path) == 0:
        print("[CMD] No valid files to merge")
        os.remove(filelist_path)
        return None

    # Get output path using atomic file naming
    merged_video_path, temp_file = get_next_filename(".mp4", output_dir=output_dir)

    # If audio is present, add proper handling with the map command to ensure all audio streams are preserved
    if has_audio:
        cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c:v copy -c:a aac -b:a 192k -map 0:v? -map 0:a? -shortest "{merged_video_path}"'
        print(f"[CMD] Merging videos with audio preservation")
    else:
        cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{merged_video_path}"'
        print(f"[CMD] Merging videos (no audio detected)")

    # Run the ffmpeg command
    try:
        result = subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
        print(f"[CMD] FFmpeg merge command completed successfully")
        # Clean up temp file after successful generation
        remove_temp_file(temp_file)
    except subprocess.CalledProcessError as e:
        print(f"[CMD] Error during video merging: {e}")
        print(f"[CMD] FFmpeg error output: {e.stderr.decode('utf-8', errors='replace') if e.stderr else 'No error output'}")
        remove_temp_file(temp_file)  # Also clean up on error
        if os.path.exists(filelist_path):
            os.remove(filelist_path)
        return None

    # Clean up the temporary file
    os.remove(filelist_path)

    # Verify the result
    if os.path.exists(merged_video_path) and os.path.getsize(merged_video_path) > 0:
        output_has_audio = check_video_has_audio(merged_video_path)
        print(f"[CMD] Successfully merged videos to {merged_video_path}")
        print(f"[CMD] Output video has audio: {output_has_audio}")

        # If we expected audio but the merged video doesn't have it, try adding it from the first video with audio
        if has_audio and not output_has_audio:
            print(f"[CMD] Audio preservation failed during merge, attempting to add audio manually")
            audio_source = next((vf for vf in video_files if os.path.exists(vf) and check_video_has_audio(vf)), None)
            if audio_source:
                merged_video_with_audio, _ = add_audio_to_video(audio_source, merged_video_path)
                if merged_video_with_audio != merged_video_path:
                    merged_video_path = merged_video_with_audio
                    print(f"[CMD] Added audio to merged video manually: {merged_video_path}")
    else:
        print(f"[CMD] Failed to merge videos or output file is empty")
        return None

    return merged_video_path

def get_common_file(new_path, old_path):
    if os.path.exists(new_path):
        return new_path
    elif os.path.exists(old_path):
        return old_path
    else:
        print(f"[WARNING] Neither {new_path} nor {old_path} found. Using {old_path} as fallback.")
        return old_path

# ------------------------- Pipeline Management Helpers -------------------------

def has_model_config_changed(old_config, new_config):
    critical_keys = ["model_choice", "torch_dtype", "num_persistent"]
    for key in critical_keys:
        if str(old_config.get(key)) != str(new_config.get(key)):
            print(f"[CMD - DEBUG] Critical config change detected in {key}: {old_config.get(key)} != {new_config.get(key)}")
            return True

    lora_keys = [
        "lora_model", "lora_alpha",
        "lora_model_2", "lora_alpha_2",
        "lora_model_3", "lora_alpha_3",
        "lora_model_4", "lora_alpha_4"
    ]

    for key in lora_keys:
        old_val = old_config.get(key)
        new_val = new_config.get(key)
        if str(old_val).strip() in ["", "None"] and str(new_val).strip() in ["", "None"]:
            continue
        if str(old_val) != str(new_val):
            print(f"[CMD - DEBUG] LoRA config change detected in {key}: {old_val} != {new_val}")
            return True
    return False

def clear_pipeline_if_needed(pipeline, pipeline_config, new_config):
    global model_manager

    if pipeline is not None and has_model_config_changed(pipeline_config, new_config):
        print(f"[CMD - DEBUG] Pipeline config changed. Clearing pipeline.")
        try:
            del pipeline
        except Exception as e:
            print(f"[CMD] Error deleting pipeline: {e}")
        pipeline = None
        pipeline_config = {}

        if model_manager is not None and hasattr(model_manager, 'clear_models'):
            model_manager.clear_models()
        try:
            del model_manager
            model_manager = None
        except Exception as e:
            print(f"[CMD] Error deleting model_manager: {e}")

        if model_manager is None:
            model_manager = ModelManager(device="cpu")

        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        print("[CMD] Pipeline and Model Manager cleared due to model config change.")
    return pipeline, pipeline_config

# ------------------------- Resolution and VRAM presets -------------------------

ASPECT_RATIOS_1_3b = {
    "1:1":  (640, 640),
    "4:3":  (736, 544),
    "3:4":  (544, 736),
    "3:2":  (768, 512),
    "2:3":  (512, 768),
    "16:9": (832, 480),
    "9:16": (480, 832),
    "21:9": (960, 416),
    "9:21": (416, 960),
    "4:5":  (560, 704),
    "5:4":  (704, 560),
}

ASPECT_RATIOS_14b = {
    "1:1":  (960, 960),
    "4:3":  (1104, 832),
    "3:4":  (832, 1104),
    "3:2":  (1152, 768),
    "2:3":  (768, 1152),
    "16:9": (1280, 720),
    "16:9_low": (832, 480),
    "9:16": (720, 1280),
    "9:16_low": (480, 832),
    "21:9": (1472, 624),
    "9:21": (624, 1472),
    "4:5":  (864, 1072),
    "5:4":  (1072, 864),
}

def aspect_choices_for_model(model_choice):
    if model_choice in [MODEL_T2V_14B, MODEL_I2V_720P]:
        return list(ASPECT_RATIOS_14b.keys())
    return list(ASPECT_RATIOS_1_3b.keys())

def update_vram_and_resolution(model_choice, preset, torch_dtype):
    if torch_dtype == "torch.float8_e4m3fn":
        if model_choice == "WAN 2.1 14B Text-to-Video":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "8,750,000,000",
                "32GB": "22,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "6,000,000,000",
                "32GB": "14,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "2,500,000,000",
                "16GB": "7,500,000,000",
                "24GB": "15,000,000,000",
                "32GB": "22,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
            default_aspect = "16:9"
        else:
            if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
                mapping = {
                    "4GB": "0",
                    "6GB": "500,000,000",
                    "8GB": "7,000,000,000",
                    "10GB": "7,000,000,000",
                    "12GB": "7,000,000,000",
                    "16GB": "7,000,000,000",
                    "24GB": "7,000,000,000",
                    "32GB": "7,000,000,000",
                    "48GB": "7,000,000,000",
                    "80GB": "7,000,000,000"
                }
                resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
                default_aspect = "16:9"
            else:
                mapping = {
                    "4GB": "0",
                    "6GB": "0",
                    "8GB": "0",
                    "10GB": "0",
                    "12GB": "0",
                    "16GB": "0",
                    "24GB": "3,000,000,000",
                    "32GB": "6,500,000,000",
                    "48GB": "16,000,000,000",
                    "80GB": "22,000,000,000"
                }
                resolution_choices = list(ASPECT_RATIOS_14b.keys())
                default_aspect = "16:9"
        return mapping.get(preset, "12000000000"), resolution_choices, default_aspect
    else:
        if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
            mapping = {
                "4GB": "0",
                "6GB": "500,000,000",
                "8GB": "7,000,000,000",
                "10GB": "7,000,000,000",
                "12GB": "7,000,000,000",
                "16GB": "7,000,000,000",
                "24GB": "7,000,000,000",
                "32GB": "7,000,000,000",
                "48GB": "7,000,000,000",
                "80GB": "7,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Text-to-Video":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "4,250,000,000",
                "32GB": "6,500,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "3,000,000,000",
                "32GB": "5,500,000,000",
                "48GB": "14,500,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "1,500,000,000",
                "16GB": "3,500,000,000",
                "24GB": "7,000,000,000",
                "32GB": "10,500,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
            default_aspect = "16:9"
        else:
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "3,000,000,000",
                "32GB": "5,500,000,000",
                "48GB": "16,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        return mapping.get(preset, "12000000000"), resolution_choices, default_aspect

def detect_vram_preset():
    """The largest VRAM preset that fits the first GPU, used for the default config."""
    try:
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            best = VRAM_PRESETS[0]
            for preset in VRAM_PRESETS:
                if int(preset[:-2]) <= total_gb + 0.75:
                    best = preset
            return best
    except Exception as e:
        print(f"[CMD] Could not detect the GPU VRAM: {e}")
    return "24GB"

DEFAULT_VRAM_PRESET = detect_vram_preset()

# ------------------------- Attention -------------------------

def default_attention_choice():
    available = wan_video_dit.available_attention_backends()
    if available["sage"]:
        return ATTENTION_SAGE
    if available["flash"]:
        return ATTENTION_FLASH
    return ATTENTION_SDPA

def apply_attention_choice(choice):
    """Select the DiT attention kernel and RoPE precision; returns the backend that will run."""
    backend, rope = ATTENTION_CHOICES.get(choice, ATTENTION_CHOICES[default_attention_choice()])
    wan_video_dit.set_attention_backend(backend)
    wan_video_dit.set_rope_precision(rope)
    return wan_video_dit.resolve_attention_backend()

# ------------------------- Configuration Management -------------------------

CONFIG_DIR = "configs"
LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config.txt")
DEFAULT_CONFIG_NAME = "Default"

# Every setting a config stores, in the order of the UI components they are loaded into
CONFIG_KEYS = [
    "model_choice", "vram_preset", "aspect_ratio", "width", "height", "auto_crop", "auto_scale", "tiled",
    "inference_steps", "pr_rife", "pr_rife_multiplier", "cfg_scale", "sigma_shift", "num_persistent",
    "torch_dtype", "lora_model", "lora_alpha", "lora_model_2", "lora_alpha_2", "lora_model_3",
    "lora_alpha_3", "lora_model_4", "lora_alpha_4", "clear_cache_after_gen", "negative_prompt",
    "save_prompt", "multiline", "num_generations", "use_random_seed", "seed", "quality", "fps",
    "num_frames", "denoising_strength", "tar_lang", "batch_folder", "batch_output_folder",
    "skip_overwrite", "save_prompt_batch", "enable_teacache", "tea_cache_l1_thresh",
    "tea_cache_model_id", "extend_factor", "attention_backend", "prompt",
]

DEFAULT_NEGATIVE_PROMPT = "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down"

# Ranges of the sliders; values outside them are clamped when a config is loaded
SLIDER_RANGES = {
    "width": (320, 1536), "height": (320, 1536), "inference_steps": (1, 100), "quality": (1, 10),
    "fps": (8, 30), "num_frames": (1, 300), "cfg_scale": (1.0, 12.0), "sigma_shift": (1.0, 12.0),
    "lora_alpha": (0.1, 2.0), "lora_alpha_2": (0.1, 2.0), "lora_alpha_3": (0.1, 2.0), "lora_alpha_4": (0.1, 2.0),
    "tea_cache_l1_thresh": (0.0, 1.0), "denoising_strength": (0.0, 1.0), "extend_factor": (1, 10),
}
INT_KEYS = {"width", "height", "inference_steps", "quality", "fps", "num_frames", "extend_factor", "num_generations"}
BOOL_KEYS = {"auto_crop", "auto_scale", "tiled", "pr_rife", "clear_cache_after_gen", "save_prompt", "multiline",
             "use_random_seed", "skip_overwrite", "save_prompt_batch", "enable_teacache"}

def get_default_config():
    return {
        "model_choice": "WAN 2.1 1.3B (Text/Video-to-Video)",
        "vram_preset": DEFAULT_VRAM_PRESET,
        "aspect_ratio": "16:9",
        "width": 832,
        "height": 480,
        "auto_crop": True,
        "auto_scale": False,
        "tiled": True,
        "inference_steps": 50,
        "pr_rife": True,
        "pr_rife_multiplier": "2x FPS",
        "cfg_scale": 5.0,
        "sigma_shift": 5.6,
        "num_persistent": update_vram_and_resolution("WAN 2.1 1.3B (Text/Video-to-Video)", DEFAULT_VRAM_PRESET, "torch.bfloat16")[0],
        "torch_dtype": "torch.bfloat16",
        "lora_model": "None",
        "lora_alpha": 1.0,
        "lora_model_2": "None",
        "lora_alpha_2": 1.0,
        "lora_model_3": "None",
        "lora_alpha_3": 1.0,
        "lora_model_4": "None",
        "lora_alpha_4": 1.0,
        "clear_cache_after_gen": DEFAULT_CLEAR_CACHE,
        "prompt": "",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "save_prompt": True,
        "multiline": False,
        "num_generations": 1,
        "use_random_seed": True,
        "seed": "",
        "quality": 10,
        "fps": 16,
        "num_frames": 81,
        "denoising_strength": 0.7,
        "tar_lang": "EN",
        "batch_folder": "batch_inputs",
        "batch_output_folder": "batch_outputs",
        "skip_overwrite": True,
        "save_prompt_batch": True,
        "enable_teacache": False,
        "tea_cache_l1_thresh": 0.15,
        "tea_cache_model_id": "Wan2.1-T2V-1.3B",
        "extend_factor": 1,
        "attention_backend": default_attention_choice(),
    }

def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def sanitize_config(values):
    """Fill missing keys with defaults and bring every value into what its UI component accepts."""
    defaults = get_default_config()
    clean = dict(defaults)
    clean.update({key: value for key, value in (values or {}).items() if key in defaults})
    choice_sets = {
        "model_choice": MODEL_CHOICES,
        "vram_preset": VRAM_PRESETS,
        "torch_dtype": DTYPE_CHOICES,
        "pr_rife_multiplier": RIFE_CHOICES,
        "tar_lang": TARGET_LANGUAGES,
        "attention_backend": list(ATTENTION_CHOICES),
    }
    for key, choices in choice_sets.items():
        if clean[key] not in choices:
            clean[key] = defaults[key]
    if clean["aspect_ratio"] not in aspect_choices_for_model(clean["model_choice"]):
        clean["aspect_ratio"] = "16:9"
    for key, (low, high) in SLIDER_RANGES.items():
        value = _to_float(clean[key], _to_float(defaults[key], low))
        value = max(low, min(high, value))
        clean[key] = int(round(value)) if key in INT_KEYS else value
    clean["num_generations"] = max(1, int(_to_float(clean["num_generations"], 1)))
    for key in BOOL_KEYS:
        clean[key] = bool(clean[key])
    for key in ("num_persistent", "seed", "prompt", "negative_prompt", "tea_cache_model_id", "batch_folder", "batch_output_folder"):
        clean[key] = "" if clean[key] is None else str(clean[key])
    for key in ("lora_model", "lora_model_2", "lora_model_3", "lora_model_4"):
        clean[key] = str(clean[key]) if clean[key] else "None"
    return clean

if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

default_config = get_default_config()

def _read_config_file(name):
    with open(os.path.join(CONFIG_DIR, f"{name}.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def _write_config_file(name, data):
    with open(os.path.join(CONFIG_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def _write_last_config(name):
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(name)

last_config = DEFAULT_CONFIG_NAME
config_loaded = None
if os.path.exists(LAST_CONFIG_FILE):
    with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as f:
        last_config_name = f.read().strip()
    if last_config_name and os.path.exists(os.path.join(CONFIG_DIR, f"{last_config_name}.json")):
        try:
            config_loaded = _read_config_file(last_config_name)
            last_config = last_config_name
        except Exception as e:
            print(f"[CMD] Could not read config '{last_config_name}': {e}")
if config_loaded is None:
    default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
    config_loaded = None
    if os.path.exists(default_config_path):
        try:
            config_loaded = _read_config_file(DEFAULT_CONFIG_NAME)
        except Exception as e:
            print(f"[CMD] Could not read config '{DEFAULT_CONFIG_NAME}': {e}")
    if config_loaded is None:
        config_loaded = default_config
        _write_config_file(DEFAULT_CONFIG_NAME, config_loaded)
    last_config = DEFAULT_CONFIG_NAME
    _write_last_config(DEFAULT_CONFIG_NAME)
config_loaded = sanitize_config(config_loaded)
last_selected_aspect_ratio = config_loaded["aspect_ratio"]

def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    files = os.listdir(CONFIG_DIR)
    configs = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
    return sorted(configs)

# Helper function to format LoRA alpha values consistently
def format_alpha(alpha):
    try:
        return str(float(alpha))
    except Exception:
        return str(alpha)

def config_component_updates(values):
    """UI updates for every component in CONFIG_KEYS order."""
    lora_choices = get_lora_choices()
    updates = []
    for key in CONFIG_KEYS:
        value = values[key]
        if key == "aspect_ratio":
            updates.append(gr.update(choices=aspect_choices_for_model(values["model_choice"]), value=value))
        elif key in ("lora_model", "lora_model_2", "lora_model_3", "lora_model_4"):
            updates.append(gr.update(choices=lora_choices, value=value if value in lora_choices else "None"))
        else:
            updates.append(gr.update(value=value))
    return updates

def save_config(config_name, *values):
    config_name = (config_name or "").strip()
    unchanged = [gr.update() for _ in CONFIG_KEYS]
    if not config_name:
        return ["Config name cannot be empty", gr.update(choices=get_config_list())] + unchanged
    if re.search(r'[<>:"/\\|?*]', config_name) or config_name in (".", ".."):
        return [f"Config name '{config_name}' contains characters that cannot be used in a file name: < > : \" / \\ | ? *", gr.update(choices=get_config_list())] + unchanged

    values = dict(zip(CONFIG_KEYS, values))
    config_data = dict(values)
    config_data["prompt"] = str(values["prompt"]) if values["prompt"] is not None else ""
    for model_key, alpha_key in (("lora_model", "lora_alpha"), ("lora_model_2", "lora_alpha_2"),
                                 ("lora_model_3", "lora_alpha_3"), ("lora_model_4", "lora_alpha_4")):
        config_data[alpha_key] = format_alpha(values[alpha_key]) if values[model_key] != "None" else "None"

    try:
        _write_config_file(config_name, config_data)
        _write_last_config(config_name)
        global last_selected_aspect_ratio
        last_selected_aspect_ratio = values["aspect_ratio"]
        return [f"Config '{config_name}' saved and loaded.", gr.update(choices=get_config_list(), value=config_name)] + unchanged
    except Exception as e:
        return [f"Error saving config: {str(e)}", gr.update(choices=get_config_list())] + unchanged

def load_config(selected_config):
    global last_selected_aspect_ratio
    status = f"Config '{selected_config}' loaded."
    if not selected_config or not os.path.exists(os.path.join(CONFIG_DIR, f"{selected_config}.json")):
        status = f"Config '{selected_config}' not found."
        values = get_default_config()
    else:
        try:
            values = _read_config_file(selected_config)
            _write_last_config(selected_config)
        except Exception as e:
            status = f"Error loading config: {str(e)}"
            values = get_default_config()
    values = sanitize_config(values)
    missing_loras = [values[key] for key in ("lora_model", "lora_model_2", "lora_model_3", "lora_model_4")
                     if values[key] != "None" and values[key] not in get_lora_choices()]
    if missing_loras:
        status += f" LoRA file(s) not found in the LoRAs folder, set to None: {', '.join(missing_loras)}"
    last_selected_aspect_ratio = values["aspect_ratio"]
    return [status] + config_component_updates(values)

def reset_config_to_defaults():
    global last_selected_aspect_ratio
    values = sanitize_config(get_default_config())
    last_selected_aspect_ratio = values["aspect_ratio"]
    return ["Default settings loaded (not saved yet: enter a name and press Save to keep them)."] + config_component_updates(values)

def refresh_config_list(current):
    choices = get_config_list()
    return gr.update(choices=choices, value=current if current in choices else None)

def process_random_prompt(prompt):
    pattern = r'<random:\s*([^>]+)>'
    def replacer(match):
        options = [option.strip() for option in match.group(1).split(',') if option.strip()]
        if options:
            return random.choice(options)
        return ''
    return re.sub(pattern, replacer, prompt)

def compute_auto_scale_dimensions(image, default_width, default_height):
    target_area = default_width * default_height
    orig_w, orig_h = image.size
    if orig_w * orig_h <= target_area:
        return orig_w, orig_h
    scale_factor = (target_area / (orig_w * orig_h)) ** 0.5
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    return new_w, new_h

def update_target_dimensions(image, auto_scale, current_width, current_height):
    if auto_scale and image is not None:
        try:
            new_w, new_h = compute_auto_scale_dimensions(image, current_width, current_height)
            return new_w, new_h
        except Exception as e:
            return current_width, current_height
    return current_width, current_height

def auto_crop_image(image, target_width, target_height):
    w, h = image.size
    target_ratio = target_width / target_height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_width = int(h * target_ratio)
        left = (w - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        new_height = int(w / target_ratio)
        top = (h - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, w, bottom))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image

def auto_scale_image(image, target_width, target_height):
    target_area = target_width * target_height
    orig_w, orig_h = image.size
    if orig_w * orig_h <= target_area:
        return image
    scale_factor = (target_area / (orig_w * orig_h)) ** 0.5
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    return image.resize((new_w, new_h), Image.LANCZOS)

def toggle_lora_visibility(current_visibility):
    new_visibility = not current_visibility
    new_label = "🧩  Hide More LoRAs" if new_visibility else "🧩  Show More LoRAs"
    return gr.update(visible=new_visibility), new_visibility, new_label

def update_tea_cache_model_id(model_choice):
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
        return "Wan2.1-T2V-1.3B"
    elif model_choice == "WAN 2.1 14B Text-to-Video":
        return "Wan2.1-T2V-14B"
    elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
        return "Wan2.1-I2V-14B-720P"
    elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
        return "Wan2.1-I2V-14B-480P"
    return "Wan2.1-T2V-1.3B"

def update_model_settings(model_choice, current_vram_preset, torch_dtype):
    global last_selected_aspect_ratio

    num_persistent_val, aspect_options, default_aspect = update_vram_and_resolution(model_choice, current_vram_preset, torch_dtype)

    aspect_to_use = last_selected_aspect_ratio if last_selected_aspect_ratio else default_aspect

    # Preserve the low aspect ratio even when switching models:
    # if the aspect ratio includes "_low" suffix, try to find the base aspect ratio
    if aspect_to_use and "_low" in aspect_to_use:
        base_aspect = aspect_to_use.split("_")[0]  # Extract base aspect ratio without "_low"

        # If switching to 1.3B model from 14B with a low aspect ratio
        if (model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P"):
            # Check if base aspect exists in 1.3B options
            if base_aspect in ASPECT_RATIOS_1_3b:
                aspect_to_use = base_aspect
            else:
                aspect_to_use = default_aspect
        # Keep the "_low" version when using 14B models
        else:
            # If the low variant isn't in choices, fall back to base aspect
            if aspect_to_use not in ASPECT_RATIOS_14b:
                if base_aspect in ASPECT_RATIOS_14b:
                    aspect_to_use = base_aspect
                else:
                    aspect_to_use = default_aspect
    else:
        # Original logic for non-low aspect ratios
        if (model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P"):
            if aspect_to_use not in ASPECT_RATIOS_1_3b:
                aspect_to_use = default_aspect
        else:
            if aspect_to_use not in ASPECT_RATIOS_14b:
                aspect_to_use = default_aspect

    # Get width and height based on selected aspect ratio
    if (model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P"):
        default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_to_use, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(aspect_to_use, (1280, 720))

    return (
        gr.update(choices=aspect_options, value=aspect_to_use),
        default_width,
        default_height,
        num_persistent_val
    )

def update_width_height(aspect_ratio, model_choice):
    global last_selected_aspect_ratio
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        # For 1.3B models, we need to remove "_low" suffix if present, since it's only valid for 14B models
        if aspect_ratio and "_low" in aspect_ratio:
            base_aspect = aspect_ratio.split("_")[0]
            if base_aspect in ASPECT_RATIOS_1_3b:
                aspect_ratio = base_aspect
            else:
                aspect_ratio = "16:9"  # fallback
        elif aspect_ratio not in ASPECT_RATIOS_1_3b:
            aspect_ratio = "16:9"  # fallback to default if invalid for 1.3B
    else:
        # For 14B models, preserve the low aspect ratio if it exists
        if aspect_ratio not in ASPECT_RATIOS_14b:
            # If it has "_low" suffix but not in dictionary
            if aspect_ratio and "_low" in aspect_ratio:
                base_aspect = aspect_ratio.split("_")[0]
                if base_aspect in ASPECT_RATIOS_14b:
                    aspect_ratio = base_aspect
                else:
                    aspect_ratio = "16:9"  # fallback
            else:
                aspect_ratio = "16:9"  # fallback

    last_selected_aspect_ratio = aspect_ratio

    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_ratio, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(aspect_ratio, (1280, 720))

    return default_width, default_height

# ------------------------- Prompt Enhance -------------------------

def resolve_prompt_extend_model():
    """Local Qwen folder from the model downloader first, else the Hugging Face model (cache)."""
    if args is not None and args.prompt_extend_model:
        return args.prompt_extend_model
    if args is None or args.prompt_extend_method == "local_qwen":
        if os.path.isfile(os.path.join(QWEN_LOCAL_DIR, "config.json")):
            return QWEN_LOCAL_DIR
    return None

def unload_prompt_expander():
    """Free the prompt enhance LLM (about 10 GB VRAM) before a video model is loaded."""
    global prompt_expander
    if prompt_expander is None:
        return
    print("[CMD] Unloading the Prompt Enhance model to free VRAM for video generation.")
    for attribute in ("model", "tokenizer", "processor"):
        if hasattr(prompt_expander, attribute):
            try:
                delattr(prompt_expander, attribute)
            except Exception:
                pass
    prompt_expander = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def prompt_enc(prompt, tar_lang):
    global prompt_expander, loaded_pipeline, loaded_pipeline_config, args
    if not prompt or not str(prompt).strip():
        gr.Warning("Write a prompt first, then press Prompt Enhance.")
        return prompt
    try:
        if prompt_expander is None:
            model_name = resolve_prompt_extend_model()
            if args.prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(model_name=model_name, is_vl=False)
            elif args.prompt_extend_method == "local_qwen":
                print(f"[CMD] Loading the Prompt Enhance model: {model_name or 'Qwen/Qwen2.5-14B-Instruct'}")
                prompt_expander = QwenPromptExpander(model_name=model_name, is_vl=False, device=0)
            else:
                raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")
        prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
        if not prompt_output.status:
            gr.Warning(f"Prompt Enhance failed: {prompt_output.message}")
        result = prompt if not prompt_output.status else prompt_output.prompt
        return result
    except Exception as e:
        traceback.print_exc()
        unload_prompt_expander()
        gr.Warning(f"Prompt Enhance failed: {e}. Download Qwen2.5-14B-Instruct with the model downloader (option 5).")
        return prompt

def show_extension_info():
    info = (
        "**Extended Video Feature – Detailed Explanation:**\n\n"
        "**Purpose:**\n"
        "- The 'Extend Video Factor' slider enables you to automatically lengthen your generated video by appending extra segments. These segments are generated using the last frame of the previous segment to maintain continuity.\n\n"
        "**How It Works:**\n"
        "1. **Initial Generation:**\n"
        "   - If the slider is set to **1×**, no extension is applied and only a single video segment is generated.\n"
        "   - For values greater than **1×**, the app first produces the initial video segment normally.\n\n"
        "2. **Determining the Number of Segments:**\n"
        "   - For **image inputs**: The total number of segments equals the slider value minus 1.\n"
        "   - For **video inputs**: Similar logic applies (with some adjustments internally).\n\n"
        "3. **Extension Generation Process:**\n"
        "   - For each additional extension, the app extracts the last frame from the most recent segment using OpenCV.\n"
        "   - This last frame is then re-fed to the generation pipeline using almost the same parameters, ensuring consistency.\n"
        "   - Text-to-Video models continue with the matching Image-to-Video model (1.3B uses 14B I2V 480P, 14B T2V uses 14B I2V 720P), so those models must be downloaded too.\n\n"
        "4. **Merging Segments:**\n"
        "   - All generated segments, including the base segment and all extensions, are merged together into one final video using ffmpeg.\n\n"
        "5. **Optional Frame-Rate Enhancement:**\n"
        "   - If the Practical-RIFE option is enabled, the final merged video undergoes frame-rate enhancement for smoother motion.\n\n"
        "6. **Batch Processing:**\n"
        "   - When processing a folder of files, a similar extension process is applied.\n\n"
        "By clicking this button, you get full insight into what the app does behind the scenes."
    )
    return info

# ------------------------- Single Generation Pipeline (Improved) -------------------------

def get_next_generation_number(output_folder):
    # Create a lock file in a temporary directory to ensure atomic operations
    lock_file = os.path.join(tempfile.gettempdir(), "wan21_generation_lock.lock")
    lock = FileLock(lock_file, timeout=10)  # 10 seconds timeout

    with lock:
        max_num = 0
        if os.path.exists(output_folder):
            for f in os.listdir(output_folder):
                m = re.match(r'^(\d{4})\.mp4$', f)
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num

        # Add a small random delay to further reduce collision chance
        time.sleep(random.uniform(0.01, 0.05))
        return max_num + 1

def pipeline_progress(progress, desc):
    """Denoising progress: the tqdm bar in the console (as before) and the progress bar in the browser."""
    def progress_bar_cmd(iterable):
        total = len(iterable)
        for index, item in enumerate(tqdm(iterable, desc=desc)):
            if progress is not None:
                try:
                    progress((index, total), desc=f"{desc}: step {index + 1}/{total}")
                except Exception:
                    pass
            yield item
    return progress_bar_cmd

def missing_model_files(model_choice):
    """Files of a model that are not on disk (shared files may also sit in the model folder)."""
    folder, shards = MODEL_FILES[model_choice]
    model_dir = os.path.join(MODELS_DIR, "Wan-AI", folder)
    missing = [os.path.join(model_dir, shard) for shard in shards if not os.path.exists(os.path.join(model_dir, shard))]
    shared = [T5_FILE, VAE_FILE] + ([CLIP_FILE] if model_choice.startswith("14B_image") else [])
    for name in shared:
        if not os.path.exists(os.path.join(MODELS_DIR, name)) and not os.path.exists(os.path.join(model_dir, name)):
            missing.append(os.path.join(MODELS_DIR, name))
    if not os.path.isdir(os.path.join(MODELS_DIR, "google", "umt5-xxl")) and not os.path.isdir(os.path.join(model_dir, "google", "umt5-xxl")):
        missing.append(os.path.join(MODELS_DIR, "google", "umt5-xxl"))
    return missing

def ensure_model_downloaded(model_choice):
    missing = missing_model_files(model_choice)
    if missing:
        option = MODEL_DOWNLOAD_OPTION[model_choice]
        raise FileNotFoundError(
            f"The {MODEL_FILES[model_choice][0]} model is not downloaded (missing: {', '.join(missing[:3])}"
            f"{' ...' if len(missing) > 3 else ''}). Run Windows_Download_Models.bat (or Download_Ubuntu.sh) and select option {option}."
        )

def generate_videos(
    prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
    aspect_ratio, width, height, auto_crop, auto_scale, tiled,
    inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
    enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
    lora_model, lora_alpha,
    lora_model_2, lora_alpha_2,
    lora_model_3, lora_alpha_slider_3,
    lora_model_4, lora_alpha_4,
    clear_cache_after_gen, extend_factor,
    attention_choice=None,
    progress=gr.Progress(),
    override_input_file=None,
    output_dir_override=None,
    custom_output_filename=None
):
    global loaded_pipeline, loaded_pipeline_config, cancel_flag, prompt_expander

    output_folder = output_dir_override or DEFAULT_OUTPUT_DIR
    # An emptied number box arrives as None
    num_generations = max(1, int(num_generations or 1))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if input_image is None and input_video is None and override_input_file is not None:
        ext = os.path.splitext(override_input_file)[1].lower()
        if ext == ".mp4":
            input_video = override_input_file
        else:
            try:
                loaded_img = Image.open(override_input_file)
                loaded_img = ImageOps.exif_transpose(loaded_img)
                input_image = loaded_img.convert("RGB")
            except Exception as e:
                print(f"[CMD] Error loading file {override_input_file}: {e}")

    cancel_flag = False
    log_text = ""
    last_used_seed = None
    overall_start_time = time.time()
    final_output_video = None

    # Attention kernel for this generation (no model reload needed)
    attention_used = ATTENTION_NAMES[apply_attention_choice(attention_choice or default_attention_choice())]
    log_text += f"[CMD] Attention: {attention_used}\n"

    input_was_video = False
    orig_video_path = None
    # Extract audio once from the original input video
    temp_audio_file = None

    if input_image is None and input_video is not None:
        input_was_video = True
        orig_video_path = input_video if isinstance(input_video, str) else input_video.name
        log_text += f"[CMD] Using input video: {orig_video_path}\n"

        # Extract audio from original video once upfront
        if orig_video_path:
            has_audio = check_video_has_audio(orig_video_path)
            if has_audio:
                log_text += f"[CMD] Input video has audio. Extracting audio once for reuse.\n"
                timestamp = int(time.time())
                temp_dir = "temp_videos"
                os.makedirs(temp_dir, exist_ok=True)
                temp_audio_file = os.path.join(temp_dir, f"temp_audio_{timestamp}.aac")

                # Arguments as a list, so paths with spaces or quotes work
                extract_cmd = [
                    'ffmpeg', '-y',
                    '-i', orig_video_path,
                    '-vn',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-v', 'info',
                    temp_audio_file
                ]

                log_text += f"[CMD] Extracting audio with command: {' '.join(extract_cmd)}\n"
                result = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

                # Verify the audio file was created successfully
                if os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                    log_text += f"[CMD] Successfully extracted audio to {temp_audio_file} (size: {os.path.getsize(temp_audio_file)} bytes)\n"
                else:
                    log_text += f"[CMD] Failed to extract audio or audio file is empty\n"
                    temp_audio_file = None
            else:
                log_text += f"[CMD] Input video has no audio to extract.\n"

        # Note: We'll re-encode the video later after effective_num_frames is defined

    if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
        model_choice = "1.3B"
        d = ASPECT_RATIOS_1_3b
    elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
        model_choice = "14B_text"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 480P":
        model_choice = "14B_image_480p"
        d = ASPECT_RATIOS_1_3b
    else:
        return None, "Invalid model choice.", ""

    target_width = int(width)
    target_height = int(height)

    if model_choice in ["14B_image_720p", "14B_image_480p"]:
        if input_image is None:
            if input_video is not None:
                video_path = input_video if isinstance(input_video, str) else input_video.name
                original_image = extract_last_frame(video_path)
                if original_image is None:
                    err_msg = "[CMD] Error: Could not extract image from provided video. Please upload a valid input image."
                    if clear_cache_after_gen:
                        loaded_pipeline = None
                        loaded_pipeline_config = {}
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    return None, err_msg, str(last_used_seed or "")
                log_text += "[CMD] Extracted last frame from input video for image-to-video generation.\n"
            else:
                err_msg = "[CMD] Error: Image model selected but no image provided. Please upload input image."
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return None, err_msg, str(last_used_seed or "")
        else:
            original_image = input_image.copy()
    elif auto_crop or auto_scale:
        if input_image is not None:
            original_image = input_image.copy()
        else:
            original_image = None

    # Define effective_num_frames before any potential re-encoding
    effective_num_frames = int(num_frames)

    if model_choice == "1.3B" and input_video is not None:
        original_video_path = input_video if isinstance(input_video, str) else input_video.name
        cap = cv2.VideoCapture(original_video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_value = cap.get(cv2.CAP_PROP_FPS)
            effective_num_frames = min(int(num_frames), total_frames)
            print(f"[CMD] Detected input video frame count: {total_frames}, using effective frame count: {effective_num_frames}")

            # Calculate target duration based on num_frames at 16fps
            target_duration = (effective_num_frames - 1) / 16
            print(f"[CMD] Target duration for the processed video: {target_duration:.2f} seconds")

            # Re-encode the input video to 16 FPS for video-to-video use
            if input_was_video:
                log_text += f"[CMD] Processing video-to-video with 1.3B model, checking if re-encoding to 16 FPS is needed...\n"
                reencoded_video = reencode_video_to_16fps(orig_video_path, effective_num_frames, target_width=target_width, target_height=target_height)
                if reencoded_video != orig_video_path:
                    log_text += f"[CMD] Re-encoded input video to 16 FPS: {reencoded_video}\n"
                    # Update the input_video to use the re-encoded version
                    input_video = reencoded_video
        else:
            effective_num_frames = int(num_frames)
            print("[CMD] Could not open input video, using provided frame count")
        cap.release()
    else:
        effective_num_frames = int(num_frames)

    num_persistent_input = str(num_persistent_input).replace(",", "").replace(" ", "").replace("_", "")
    try:
        vram_value = int(num_persistent_input)
    except ValueError:
        return None, log_text + f"[CMD] Error: 'Number of Persistent Parameters In Dit' must be a whole number, got '{num_persistent_input}'.", ""
    effective_loras = []
    if lora_model and lora_model != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model), lora_alpha))
    if lora_model_2 and lora_model_2 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_2), lora_alpha_2))
    if lora_model_3 and lora_model_3 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_3), lora_alpha_slider_3))
    if lora_model_4 and lora_model_4 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_4), lora_alpha_4))

    new_config = {
         "model_choice": model_choice,
         "torch_dtype": torch_dtype,
         "num_persistent": str(vram_value),
         "lora_model": lora_model,
         "lora_alpha": format_alpha(lora_alpha) if lora_model != "None" else "None",
         "lora_model_2": lora_model_2,
         "lora_alpha_2": format_alpha(lora_alpha_2) if lora_model_2 != "None" else "None",
         "lora_model_3": lora_model_3,
         "lora_alpha_3": format_alpha(lora_alpha_slider_3) if lora_model_3 != "None" else "None",
         "lora_model_4": lora_model_4,
         "lora_alpha_4": format_alpha(lora_alpha_4) if lora_model_4 != "None" else "None",
    }
    # The prompt enhance LLM would otherwise keep about 10 GB of VRAM while the video model runs
    unload_prompt_expander()
    loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, new_config)
    if loaded_pipeline is None:
         loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
         loaded_pipeline_config = new_config

    if multi_line:
        prompts_list = [line.strip() for line in prompt.splitlines() if line.strip()]
    else:
        prompts_list = [prompt.strip()]

    total_iterations = len(prompts_list) * int(num_generations)
    if custom_output_filename:
        base_name_prefix = custom_output_filename
        counter = 0
    else:
        counter = get_next_generation_number(output_folder)

    seed_input = "" if seed_input is None else str(seed_input)
    if use_random_seed:
        base_seed = None
    else:
        try:
            base_seed = int(seed_input.strip()) if seed_input.strip() != "" else random.randint(0, 2**32 - 1)
        except:
            base_seed = random.randint(0, 2**32 - 1)

    for p in prompts_list:
        for gen in range(int(num_generations)):
            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user before starting a new video.\n"
                return None, log_text, str(last_used_seed or "")

            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = base_seed + gen if int(num_generations) > 1 else base_seed
            last_used_seed = current_seed

            if custom_output_filename:
                base_name = f"{base_name_prefix}{'' if counter == 0 else '_' + str(counter)}"
                counter += 1
            else:
                base_name = f"{counter:04d}"
                counter += 1

            log_text += f"[CMD] Generation with prompt: {p} and seed: {current_seed}\n"

            base_config = {
                "model_choice": model_choice,
                "torch_dtype": torch_dtype,
                "num_persistent": str(vram_value),
                "lora_model": lora_model,
                "lora_alpha": format_alpha(lora_alpha) if lora_model != "None" else "None",
                "lora_model_2": lora_model_2,
                "lora_alpha_2": format_alpha(lora_alpha_2) if lora_model_2 != "None" else "None",
                "lora_model_3": lora_model_3,
                "lora_alpha_3": format_alpha(lora_alpha_slider_3) if lora_model_3 != "None" else "None",
                "lora_model_4": lora_model_4,
                "lora_alpha_4": format_alpha(lora_alpha_4) if lora_model_4 != "None" else "None"
            }
            if loaded_pipeline is None or loaded_pipeline_config.get("model_choice") != model_choice:
                loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, base_config)
                if loaded_pipeline is None:
                    loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
                    loaded_pipeline_config = base_config

            common_args = {
                "prompt": process_random_prompt(p),
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(inference_steps),
                "seed": current_seed,
                "tiled": tiled,
                "width": target_width,
                "height": target_height,
                "num_frames": effective_num_frames,
                "cfg_scale": cfg_scale,
                "sigma_shift": sigma_shift,
                "progress_bar_cmd": pipeline_progress(progress, "Generating video"),
            }
            if enable_teacache:
                common_args["tea_cache_l1_thresh"] = tea_cache_l1_thresh
                common_args["tea_cache_model_id"] = tea_cache_model_id
            else:
                common_args["tea_cache_l1_thresh"] = None
                common_args["tea_cache_model_id"] = ""

            # Get the original filename with atomic file generation
            original_filename, original_temp_file = get_next_filename("mp4", output_dir=output_folder,
                                                                     custom_filename=base_name if custom_output_filename else None)
            video_start_time = time.time()

            if model_choice == "1.3B":
                if input_video is not None:
                    video_obj = VideoData(input_video if isinstance(input_video, str) else input_video.name, height=target_height, width=target_width)
                    try:
                        video_data = loaded_pipeline(
                            input_video=video_obj,
                            denoising_strength=denoising_strength,
                            **common_args,
                            cancel_fn=lambda: cancel_flag
                        )
                    finally:
                        # Release the file, so the temporary re-encoded input can be deleted afterwards
                        try:
                            video_obj.data.reader.close()
                        except Exception:
                            pass
                        del video_obj
                else:
                    video_data = loaded_pipeline(
                        **common_args,
                        cancel_fn=lambda: cancel_flag
                    )
            elif model_choice in ["14B_text"]:
                video_data = loaded_pipeline(
                    **common_args,
                    cancel_fn=lambda: cancel_flag
                )
            elif model_choice in ["14B_image_720p", "14B_image_480p"]:
                if auto_crop:
                    processed_image = auto_crop_image(original_image, target_width, target_height)
                elif auto_scale:
                    processed_image = auto_scale_image(original_image, target_width, target_height)
                else:
                    processed_image = original_image

                pre_processed_dir = "auto_pre_processed_images"
                if not os.path.exists(pre_processed_dir):
                    os.makedirs(pre_processed_dir)
                save_filename = os.path.join(pre_processed_dir, f"auto_processed_{int(time.time())}.png")
                try:
                    processed_image.save(save_filename)
                    print(f"[CMD] Auto processed image saved to: {save_filename}")
                except Exception as e:
                    print(f"[CMD] Failed to save auto processed image: {e}")

                video_data = loaded_pipeline(
                    input_image=processed_image,
                    **common_args,
                    cancel_fn=lambda: cancel_flag
                )
            else:
                err_msg = "[CMD] Invalid combination of inputs."
                # Clean up any temporary file before exiting with error
                remove_temp_file(original_temp_file)
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return None, err_msg, str(last_used_seed or "")

            video_duration = time.time() - video_start_time
            log_text += f"[CMD] Original video generation duration: {video_duration:.2f} seconds\n"

            if cancel_flag and not video_data:
                remove_temp_file(original_temp_file)
                log_text += "[CMD] Generation cancelled by user during denoising.\n"
                return None, log_text, str(last_used_seed or "")

            # Send both the filename and the temp file for proper cleanup
            success, _ = safe_save_video(video_data, (original_filename, original_temp_file), fps=fps, quality=quality)
            if not success:
                # Make sure temp file is gone if save failed
                remove_temp_file(original_temp_file)
                return None, log_text + "[CMD] Failed to save video.", str(last_used_seed or "")

            log_text += f"[CMD] Saved original video: {original_filename}\n"

            # Transfer audio from input video if available
            if input_was_video and orig_video_path:
                # Use our pre-extracted audio file if available
                original_filename_with_audio, _ = add_audio_to_video(orig_video_path, original_filename, temp_audio_file=temp_audio_file)
                if original_filename_with_audio != original_filename:
                    original_filename = original_filename_with_audio
                    log_text += f"[CMD] Added audio to video: {original_filename}\n"

            if save_prompt:
                txt_filename = os.path.splitext(original_filename)[0] + ".txt"
                generation_details = generate_prompt_info({
                    "prompt": p,
                    "negative_prompt": negative_prompt,
                    "model_choice": model_choice_radio,
                    "inference_steps": inference_steps,
                    "cfg_scale": cfg_scale,
                    "sigma_shift": sigma_shift,
                    "seed": current_seed,
                    "num_frames": effective_num_frames,
                    "extend_factor": extend_factor,
                    "num_segments": 1,
                    "extension_segment": 0,
                    "total_extensions": int(extend_factor) - 1,
                    "source_frame": "original",
                    "input_file": orig_video_path if input_was_video else "",
                    "is_video": input_was_video,
                    "has_input_video": input_was_video,
                    "denoising_strength": denoising_strength,
                    "is_text_to_video": model_choice == "14B_text",
                    "lora_details": [f"{os.path.basename(path)} (scale {alpha})" for path, alpha in effective_loras],
                    "enable_teacache": enable_teacache,
                    "tea_cache_l1_thresh": tea_cache_l1_thresh,
                    "tea_cache_model_id": tea_cache_model_id,
                    "torch_dtype": torch_dtype,
                    "attention": attention_used,
                    "auto_crop": auto_crop,
                    "width": target_width,
                    "height": target_height,
                    "video_generation_duration": video_duration,
                    "generation_duration": time.time() - overall_start_time,
                    "include_minutes": True
                })
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(generation_details)
                log_text += f"[CMD] Saved prompt info for original video: {txt_filename}\n"

            # Only switch the pipeline to extension mode if extend_factor > 1
            if int(extend_factor) > 1:
                extension_model_choice = model_choice
                if model_choice == "1.3B":
                    extension_model_choice = "14B_image_480p"
                elif model_choice == "14B_text":
                    extension_model_choice = "14B_image_720p"
                if extension_model_choice != model_choice:
                    log_text += f"[CMD] Switching pipeline for extension segments to model {extension_model_choice}\n"
                    new_config_ext = new_config.copy()
                    new_config_ext["model_choice"] = extension_model_choice
                    loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, new_config_ext)
                    if loaded_pipeline is None:
                        loaded_pipeline = load_wan_pipeline(extension_model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
                        loaded_pipeline_config = new_config_ext

            original_improved = None
            ext_segments = []
            ext_segments_improved = []

            additional_extensions = int(extend_factor) - 1
            prev_video = original_filename
            for ext_iter in range(1, additional_extensions + 1):
                if cancel_flag:
                    log_text += "[CMD] Generation cancelled by user during extensions.\n"
                    break
                last_frame = extract_last_frame(prev_video)
                if last_frame is None:
                    log_text += f"[CMD] Failed to extract last frame for extension {ext_iter} from {prev_video}.\n"
                    break
                used_folder = "used_last_frames"
                if not os.path.exists(used_folder):
                    os.makedirs(used_folder)
                last_frame_filename = os.path.join(used_folder, f"{base_name}_ext{ext_iter}_lastframe.png")
                last_frame.save(last_frame_filename)
                log_text += f"[CMD] Saved last frame used for extension {ext_iter}: {last_frame_filename}\n"
                new_width, new_height = last_frame.size
                common_args_ext = {
                    "prompt": process_random_prompt(p),
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": int(inference_steps),
                    "seed": random.randint(0, 2**32 - 1) if use_random_seed else (current_seed),
                    "tiled": tiled,
                    "width": new_width,
                    "height": new_height,
                    "num_frames": int(num_frames),
                    "cfg_scale": cfg_scale,
                    "sigma_shift": sigma_shift,
                    "progress_bar_cmd": pipeline_progress(progress, f"Generating extension {ext_iter}/{additional_extensions}"),
                }
                if enable_teacache:
                    common_args_ext["tea_cache_l1_thresh"] = tea_cache_l1_thresh
                    common_args_ext["tea_cache_model_id"] = tea_cache_model_id
                else:
                    common_args_ext["tea_cache_l1_thresh"] = None
                    common_args_ext["tea_cache_model_id"] = ""

                # Get extension filename with atomic file generation
                ext_file_prefix = f"{base_name}_ext{ext_iter}_original"
                extension_filename, extension_temp_file = get_next_filename("mp4", output_dir=output_folder,
                                                                         custom_filename=ext_file_prefix)
                log_text += f"[CMD] Generating extension segment {ext_iter} (as {os.path.basename(extension_filename)}) using model {extension_model_choice if int(extend_factor)>1 else model_choice}\n"
                try:
                    ext_start_time = time.time()

                    video_data_ext = loaded_pipeline(
                        input_image=last_frame,
                        **common_args_ext,
                        cancel_fn=lambda: cancel_flag
                    )

                    ext_duration = time.time() - ext_start_time
                    log_text += f"[CMD] Extension segment {ext_iter} generation duration: {ext_duration:.2f} seconds\n"

                    if not video_data_ext:
                        log_text += "[CMD] Extension generation returned no data.\n"
                        # Clean up temp file if generation failed
                        remove_temp_file(extension_temp_file)
                        break
                    success, _ = safe_save_video(video_data_ext, (extension_filename, extension_temp_file), fps=fps, quality=quality)
                    if not success:
                        # Make sure temp file is gone if save failed
                        remove_temp_file(extension_temp_file)
                        log_text += f"[CMD] Failed to save extension segment {ext_iter}.\n"
                        break
                    log_text += f"[CMD] Saved extension segment {ext_iter}: {extension_filename}\n"

                    # For extension videos, we can also transfer audio if it's available
                    if input_was_video and orig_video_path:
                        extension_filename_with_audio, _ = add_audio_to_video(orig_video_path, extension_filename, temp_audio_file=temp_audio_file)
                        if extension_filename_with_audio != extension_filename:
                            extension_filename = extension_filename_with_audio
                            log_text += f"[CMD] Added audio to extension video: {extension_filename}\n"

                    if save_prompt:
                        txt_filename_ext = os.path.splitext(extension_filename)[0] + ".txt"
                        generation_details_ext = generate_prompt_info({
                            "prompt": p,
                            "negative_prompt": negative_prompt,
                            "model_choice": extension_model_choice if int(extend_factor) > 1 else model_choice_radio,
                            "extension_model": extension_model_choice if int(extend_factor) > 1 and extension_model_choice != model_choice_radio else None,
                            "inference_steps": inference_steps,
                            "cfg_scale": cfg_scale,
                            "sigma_shift": sigma_shift,
                            "seed": common_args_ext["seed"],
                            "num_frames": num_frames,
                            "extend_factor": extend_factor,
                            "num_segments": 1,
                            "extension_segment": ext_iter,
                            "total_extensions": additional_extensions,
                            "source_frame": os.path.basename(prev_video),
                            "input_file": orig_video_path if input_was_video else "",
                            "is_video": input_was_video,
                            "has_input_video": input_was_video,
                            "denoising_strength": denoising_strength,
                            "is_text_to_video": model_choice=="14B_text",
                            "lora_details": [f"{os.path.basename(path)} (scale {alpha})" for path, alpha in effective_loras],
                            "enable_teacache": enable_teacache,
                            "tea_cache_l1_thresh": tea_cache_l1_thresh,
                            "tea_cache_model_id": tea_cache_model_id,
                            "torch_dtype": torch_dtype,
                            "attention": attention_used,
                            "auto_crop": auto_crop,
                            "width": new_width,
                            "height": new_height,
                            "video_generation_duration": ext_duration,
                            "generation_duration": time.time() - overall_start_time,
                            "include_minutes": True
                        })
                        with open(txt_filename_ext, "w", encoding="utf-8") as f:
                            f.write(generation_details_ext)
                        log_text += f"[CMD] Saved prompt info for extension segment {ext_iter}: {txt_filename_ext}\n"
                    ext_segments.append(extension_filename)
                    prev_video = extension_filename
                except Exception as e:
                    log_text += f"[CMD] Error during extension generation: {str(e)}\n"
                    continue

            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user, skipping post-processing steps.\n"
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return original_filename, log_text, str(last_used_seed or "")

            if pr_rife_enabled:
                if cancel_flag:
                    log_text += "[CMD] Generation cancelled by user before Practical-RIFE processing.\n"
                else:
                    # Determine RIFE suffix before getting the filename
                    multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                    rife_suffix = f"_{multiplier_val}xFPS"

                    # Use base_name and the RIFE suffix for the improved filename
                    original_improved_name = f"{base_name}{rife_suffix}"
                    original_improved, original_improved_temp = get_next_filename("mp4", output_dir=output_folder, custom_filename=original_improved_name)

                    try:
                        cap = cv2.VideoCapture(original_filename)
                        source_fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        if source_fps <= 125:
                            print(f"[CMD] Applying Practical-RIFE on original {original_filename}")
                            cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{os.path.abspath(os.path.join("Practical-RIFE", "train_log"))}" --multi={multiplier_val} --video="{original_filename}" --output="{original_improved}"'
                            if not cancel_flag:
                                subprocess.run(cmd, shell=True, check=True, env=os.environ)
                                log_text += f"[CMD] Applied Practical-RIFE on original. Saved as: {original_improved}\n"
                                # Clean up temp file after successful RIFE processing
                                remove_temp_file(original_improved_temp)

                                # Re-add audio after RIFE processing if the original video had audio
                                if input_was_video and orig_video_path:
                                    original_improved_with_audio, _ = add_audio_to_video(original_filename, original_improved, temp_audio_file=temp_audio_file)
                                    if original_improved_with_audio != original_improved:
                                        original_improved = original_improved_with_audio
                                        log_text += f"[CMD] Added audio back after RIFE processing: {original_improved}\n"
                            else:
                                log_text += "[CMD] Generation cancelled by user during Practical-RIFE processing.\n"
                                original_improved = original_filename
                                # Clean up temp file if cancelled
                                remove_temp_file(original_improved_temp)
                        else:
                            original_improved = original_filename
                            log_text += f"[CMD] Skipped Practical-RIFE on original because source FPS ({source_fps:.2f}) is above threshold.\n"
                            # Clean up unused temp file
                            remove_temp_file(original_improved_temp)
                    except Exception as e:
                        log_text += f"[CMD] Error applying Practical-RIFE on original: {str(e)}\n"
                        original_improved = original_filename
                        # Clean up temp file on error
                        remove_temp_file(original_improved_temp)

                    for idx, ext_file in enumerate(ext_segments):
                        if cancel_flag:
                            log_text += f"[CMD] Generation cancelled by user before Practical-RIFE processing on extension {idx+1}.\n"
                            ext_segments_improved.append(ext_file)
                            continue

                        ext_improved_name = f"{base_name}_ext{idx+1}_original{rife_suffix}"
                        ext_improved, ext_improved_temp = get_next_filename("mp4", output_dir=output_folder, custom_filename=ext_improved_name)
                        try:
                            cap = cv2.VideoCapture(ext_file)
                            source_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            if source_fps <= 29:
                                print(f"[CMD] Applying Practical-RIFE on extension {ext_file}")
                                cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{os.path.abspath(os.path.join("Practical-RIFE", "train_log"))}" --multi={multiplier_val} --video="{ext_file}" --output="{ext_improved}"'
                                if not cancel_flag:
                                    subprocess.run(cmd, shell=True, check=True, env=os.environ)
                                    log_text += f"[CMD] Applied Practical-RIFE on extension {idx+1}. Saved as: {ext_improved}\n"
                                    # Clean up temp file after successful RIFE processing
                                    remove_temp_file(ext_improved_temp)

                                    # Re-add audio after RIFE processing if the original extension had audio
                                    ext_improved_with_audio, _ = add_audio_to_video(ext_file, ext_improved, temp_audio_file=temp_audio_file)
                                    if ext_improved_with_audio != ext_improved:
                                        ext_improved = ext_improved_with_audio
                                        log_text += f"[CMD] Added audio back after RIFE processing on extension {idx+1}: {ext_improved}\n"

                                else:
                                    log_text += f"[CMD] Generation cancelled by user during Practical-RIFE processing on extension {idx+1}.\n"
                                    ext_improved = ext_file
                                    # Clean up temp file if cancelled
                                    remove_temp_file(ext_improved_temp)
                            else:
                                ext_improved = ext_file
                                log_text += f"[CMD] Skipped Practical-RIFE on extension {idx+1} due to high FPS.\n"
                                # Clean up unused temp file
                                remove_temp_file(ext_improved_temp)
                        except Exception as e:
                            log_text += f"[CMD] Error applying Practical-RIFE on extension {idx+1}: {str(e)}\n"
                            ext_improved = ext_file
                            # Clean up temp file on error
                            remove_temp_file(ext_improved_temp)
                        ext_segments_improved.append(ext_improved)
            else:
                original_improved = original_filename

            merged_original = None
            merged_enhanced = None
            if ext_segments and not cancel_flag:
                merged_original_temp = None
                try:
                    merge_list = [original_filename] + ext_segments
                    merged_original_name = f"{base_name}_extended_{additional_extensions}"
                    merged_original, merged_original_temp = get_next_filename("mp4", output_dir=output_folder, custom_filename=merged_original_name)
                    filelist_path = os.path.join(tempfile.gettempdir(), "filelist_original.txt")
                    with open(filelist_path, "w", encoding="utf-8") as f:
                        for vf in merge_list:
                            if os.path.exists(vf):
                                f.write(f"file '{os.path.abspath(vf)}'\n")
                            else:
                                log_text += f"[CMD] Warning: file not found: {vf}\n"
                    if os.path.getsize(filelist_path) > 0 and not cancel_flag:
                        cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{merged_original}"'
                        subprocess.run(cmd, shell=True, check=True)
                        # Clean up temp file after successful merge
                        remove_temp_file(merged_original_temp)
                        os.remove(filelist_path)
                        log_text += f"[CMD] Merged unenhanced extended video saved as: {merged_original}\n"

                        # Add audio to the merged video if any of the original videos had audio
                        has_audio = any(check_video_has_audio(vf) for vf in merge_list if os.path.exists(vf))
                        if has_audio:
                            # Use the first video with audio as the source
                            audio_source = next((vf for vf in merge_list if os.path.exists(vf) and check_video_has_audio(vf)), None)
                            if audio_source:
                                merged_original_with_audio, _ = add_audio_to_video(audio_source, merged_original, temp_audio_file=temp_audio_file)
                                if merged_original_with_audio != merged_original:
                                    merged_original = merged_original_with_audio
                                    log_text += f"[CMD] Added audio to merged original video: {merged_original}\n"
                    else:
                        if cancel_flag:
                            log_text += "[CMD] Generation cancelled by user before merging original files.\n"
                        else:
                            log_text += f"[CMD] No valid files to merge for extended original.\n"
                        os.remove(filelist_path)
                        # Clean up temp file if not used for merge
                        remove_temp_file(merged_original_temp)
                except Exception as e:
                    log_text += f"[CMD] Error merging original extensions: {str(e)}\n"
                    # Clean up temp file on error
                    remove_temp_file(merged_original_temp)
                if pr_rife_enabled and ext_segments_improved and not cancel_flag:
                    merged_enhanced_temp = None
                    try:
                        merge_list_improved = [original_improved] + ext_segments_improved
                        merged_enhanced_name = f"{base_name}_extended_{additional_extensions}{rife_suffix}"
                        merged_enhanced, merged_enhanced_temp = get_next_filename("mp4", output_dir=output_folder, custom_filename=merged_enhanced_name)
                        filelist_path = os.path.join(tempfile.gettempdir(), "filelist_enhanced.txt")
                        with open(filelist_path, "w", encoding="utf-8") as f:
                            for vf in merge_list_improved:
                                if os.path.exists(vf):
                                    f.write(f"file '{os.path.abspath(vf)}'\n")
                                else:
                                    log_text += f"[CMD] Warning: file not found: {vf}\n"
                        if os.path.getsize(filelist_path) > 0 and not cancel_flag:
                            cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{merged_enhanced}"'
                            subprocess.run(cmd, shell=True, check=True)
                            # Clean up temp file after successful merge
                            remove_temp_file(merged_enhanced_temp)
                            os.remove(filelist_path)
                            log_text += f"[CMD] Merged enhanced extended video saved as: {merged_enhanced}\n"

                            # Add audio to the merged enhanced video if any of the improved videos had audio
                            has_audio = any(check_video_has_audio(vf) for vf in merge_list_improved if os.path.exists(vf))
                            if has_audio:
                                # Use the first video with audio as the source
                                audio_source = next((vf for vf in merge_list_improved if os.path.exists(vf) and check_video_has_audio(vf)), None)
                                if audio_source:
                                    merged_enhanced_with_audio, _ = add_audio_to_video(audio_source, merged_enhanced, temp_audio_file=temp_audio_file)
                                    if merged_enhanced_with_audio != merged_enhanced:
                                        merged_enhanced = merged_enhanced_with_audio
                                        log_text += f"[CMD] Added audio to merged enhanced video: {merged_enhanced}\n"
                        else:
                            if cancel_flag:
                                log_text += "[CMD] Generation cancelled by user before merging enhanced files.\n"
                            else:
                                log_text += f"[CMD] No valid files to merge for extended enhanced video.\n"
                            os.remove(filelist_path)
                            # Clean up temp file if not used for merge
                            remove_temp_file(merged_enhanced_temp)
                    except Exception as e:
                        log_text += f"[CMD] Error merging enhanced extensions: {str(e)}\n"
                        # Clean up temp file on error
                        remove_temp_file(merged_enhanced_temp)

            if pr_rife_enabled and merged_enhanced and os.path.exists(merged_enhanced):
                final_output_video = merged_enhanced
            elif merged_original and os.path.exists(merged_original):
                final_output_video = merged_original
            elif pr_rife_enabled and original_improved and os.path.exists(original_improved): # Check RIFE enabled here
                final_output_video = original_improved
            else:
                final_output_video = original_filename

            log_text += f"[CMD] Completed generation for base {base_name}.\n"

            if clear_cache_after_gen:
                loaded_pipeline = None
                loaded_pipeline_config = {}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    overall_duration = time.time() - overall_start_time
    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
    log_text += f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes). Last used seed: {last_used_seed}\n"
    print(f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds. Last used seed: {last_used_seed}")

    # Clean up temporary re-encoded videos
    clean_temp_videos()

    if clear_cache_after_gen:
        loaded_pipeline = None
        loaded_pipeline_config = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if final_output_video and os.path.exists(final_output_video):
        return final_output_video, log_text, str(last_used_seed or "")
    elif final_output_video:
        return final_output_video, log_text + "\n[CMD] Warning: Could not generate valid output video.", str(last_used_seed or "")
    else:
        return None, log_text + "\n[CMD] Warning: Could not generate valid output video.", str(last_used_seed or "")

def generate_videos_ui(*values, progress=None):
    """Generate button: runs generate_videos and reports errors in the status log instead of a dead UI."""
    try:
        return generate_videos(*values, progress=progress, output_dir_override=DEFAULT_OUTPUT_DIR)
    except Exception as e:
        traceback.print_exc()
        message = f"[CMD] Error: {e}"
        if isinstance(e, torch.OutOfMemoryError):
            message += "\n[CMD] Out of VRAM: lower 'Number of Persistent Parameters In Dit', pick a smaller VRAM preset, use FP8, enable Tiled VAE Decode or reduce resolution / frames."
        gr.Warning(str(e)[:300])
        return None, message, ""

def cancel_generation():
    global cancel_flag, cancel_batch_flag
    cancel_flag = True
    cancel_batch_flag = True
    print("[CMD] Cancel button pressed.")
    gr.Info("Cancelling: the generation stops after the current step.")
    return "Cancelling generation..."

# ------------------------- Improved Batch Processing -------------------------

def batch_process_videos(
    default_prompt, folder_path, batch_output_folder, skip_overwrite, tar_lang, negative_prompt, denoising_strength,
    use_random_seed, seed_input, quality, fps, model_choice_radio, vram_preset, num_persistent_input,
    torch_dtype, num_frames, inference_steps, aspect_ratio, width, height, auto_crop, auto_scale,
    tiled, cfg_scale, sigma_shift, save_prompt, pr_rife_enabled, pr_rife_radio, lora_model, lora_alpha,
    lora_model_2, lora_alpha_2, lora_model_3, lora_alpha_3, lora_model_4, lora_alpha_4, enable_teacache,
    tea_cache_l1_thresh, tea_cache_model_id, clear_cache_after_gen, extend_factor, num_generations,
    attention_choice=None,
    progress=gr.Progress()
):
    """Generator: the batch log updates after every file."""
    global cancel_batch_flag, cancel_flag
    cancel_batch_flag = False
    cancel_flag = False
    log_text = ""
    folder_path = (folder_path or "").strip().strip('"')
    batch_output_folder = (batch_output_folder or "").strip().strip('"') or "batch_outputs"
    if not os.path.isdir(folder_path):
        log_text += f"[CMD] Provided folder path does not exist: {folder_path}\n"
        yield log_text
        return
    if not os.path.exists(batch_output_folder):
        try:
            os.makedirs(batch_output_folder)
            log_text += f"[CMD] Created batch processing outputs folder: {batch_output_folder}\n"
        except Exception as e:
            log_text += f"[CMD] Error creating output folder {batch_output_folder}: {e}\n"
            yield log_text
            return
    files_unsorted = os.listdir(folder_path)
    # Sort files using natural sort key
    files_unsorted.sort(key=alphanum_key)

    allowed_exts = [".jpg", ".png", ".jpeg", ".mp4", ".webp"]
    # Filter *after* sorting to maintain order of allowed files
    files = [f for f in files_unsorted if os.path.splitext(f)[1].lower() in allowed_exts]
    total_files = len(files)
    log_text += f"[CMD] Found {total_files} files in folder {folder_path} (sorted naturally)\n"
    yield log_text
    batch_start_time = time.time()
    for index, file in enumerate(files, 1):
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            yield log_text
            return

        file_path = os.path.join(folder_path, file)
        base, ext = os.path.splitext(file)
        if skip_overwrite and os.path.exists(os.path.join(batch_output_folder, base + ".mp4")):
            log_text += f"[CMD] [{index}/{total_files}] Skipping {file}: {base}.mp4 already exists in {batch_output_folder}\n"
            yield log_text
            continue
        log_text += f"[CMD] [{index}/{total_files}] Processing {file}\n"
        prompt_path = os.path.join(folder_path, base + ".txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
            if prompt_content == "":
                log_text += f"[CMD] Prompt file {base+'.txt'} is empty, using default prompt.\n"
                prompt_content = default_prompt
            else:
                log_text += f"[CMD] Using prompt from {base+'.txt'} for {file}\n"
        else:
            log_text += f"[CMD] No prompt file for {file}, using default prompt.\n"
            prompt_content = default_prompt
        yield log_text

        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            yield log_text
            return

        ext_lower = ext.lower()
        if ext_lower == ".mp4":
            image_in = None
            video_in = file_path
            orig_video_path = file_path  # Save the original video path for audio transfer

            # Re-encode the video to 16 FPS only if we're doing video-to-video with the 1.3B model
            # Don't re-encode for image-to-video models that just use the last frame
            if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
                # Ensure num_frames is valid before re-encoding
                frames_to_use = int(num_frames)
                log_text += f"[CMD] Processing video-to-video with 1.3B model for {file}, checking if re-encoding needed...\n"

                reencoded_video = reencode_video_to_16fps(video_in, frames_to_use, target_width=int(width), target_height=int(height))
                if reencoded_video != video_in:
                    log_text += f"[CMD] Re-encoded input video {file} to 16 FPS: {reencoded_video}\n"
                    video_in = reencoded_video
        else:
            try:
                loaded_img = Image.open(file_path)
                loaded_img = ImageOps.exif_transpose(loaded_img)
                image_in = loaded_img.convert("RGB")
            except Exception as e:
                log_text += f"[CMD] Error loading image {file_path}: {e}\n"
                yield log_text
                continue
            video_in = None
            orig_video_path = None  # No original video for image inputs

        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            yield log_text
            return

        custom_filename = base

        print(f"[CMD] Processing batch item: {file_path}")

        # Use the original video path (if available) as the override_input_file to ensure audio is preserved
        override_file = orig_video_path if ext_lower == ".mp4" else None

        try:
            generated_video, single_log, _ = generate_videos(
                prompt_content, tar_lang, negative_prompt, image_in, video_in, denoising_strength, num_generations,
                save_prompt, False, use_random_seed, seed_input, quality, fps,
                model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
                aspect_ratio, width, height, auto_crop, auto_scale, tiled, inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
                enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
                lora_model, lora_alpha, lora_model_2, lora_alpha_2, lora_model_3, lora_alpha_3, lora_model_4, lora_alpha_4,
                clear_cache_after_gen, extend_factor,
                attention_choice,
                progress,
                override_file,
                output_dir_override=batch_output_folder,
                custom_output_filename=custom_filename
            )
            log_text += single_log
        except Exception as e:
            traceback.print_exc()
            log_text += f"[CMD] Error while processing {file}: {e}\n"
            if isinstance(e, (FileNotFoundError, torch.OutOfMemoryError)):
                log_text += "[CMD] Batch processing stopped.\n"
                yield log_text
                return
        yield log_text

        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user after file completion.\n"
            # Clean up temporary re-encoded videos
            clean_temp_videos()
            yield log_text
            return

    # Clean up temporary re-encoded videos
    clean_temp_videos()
    # Clean up any remaining temporary files in batch output folder
    cleanup_tmp_files(batch_output_folder)
    log_text += f"[CMD] Batch processing finished in {(time.time() - batch_start_time) / 60:.2f} minutes.\n"
    yield log_text

def cancel_batch_process():
    global cancel_batch_flag, cancel_flag
    cancel_batch_flag = True
    cancel_flag = True
    print("[CMD] Batch process cancel button pressed.")
    gr.Info("Cancelling the batch: it stops after the current step.")
    return "Cancelling any active generation..."

def get_next_filename(extension, output_dir=DEFAULT_OUTPUT_DIR, custom_filename=None):
    """
    Get next available filename in sequence using atomic file operations.
    Uses file locking to prevent race conditions between multiple app instances.

    If custom_filename is provided, it will attempt to use that name instead of a sequential number.

    Returns a tuple of (filename, temp_filename) where temp_filename is the temporary
    reservation file that should be cleaned up after the real file is created.
    """
    extension = extension.lstrip(".")
    os.makedirs(output_dir, exist_ok=True)

    # Create a lock file in a temporary directory
    lock_file = os.path.join(tempfile.gettempdir(), "wan21_filename_lock.lock")
    lock = FileLock(lock_file, timeout=10)  # 10 seconds timeout

    with lock:
        # If custom filename is provided, try to use it
        if custom_filename:
            base_filename = os.path.join(output_dir, f"{custom_filename}.{extension}")
            temp_filename = base_filename + ".tmp"

            # Check if the file already exists
            if not os.path.exists(base_filename) and not os.path.exists(temp_filename):
                # Create an empty file to "reserve" this filename
                with open(temp_filename, "w") as f:
                    # Add timestamp and instance info for debugging
                    f.write(f"Reserved by process {os.getpid()} at {datetime.now().isoformat()}")

                # Add a small random delay to further reduce collision chance
                time.sleep(random.uniform(0.01, 0.05))
                return base_filename, temp_filename

            # If the file exists and we have multiple generations or extensions,
            # add a counter to make it unique
            counter = 1
            while True:
                unique_filename = os.path.join(output_dir, f"{custom_filename}_{counter}.{extension}")
                temp_filename = unique_filename + ".tmp"

                if not os.path.exists(unique_filename) and not os.path.exists(temp_filename):
                    with open(temp_filename, "w") as f:
                        f.write(f"Reserved by process {os.getpid()} at {datetime.now().isoformat()}")

                    time.sleep(random.uniform(0.01, 0.05))
                    return unique_filename, temp_filename
                counter += 1

        # Default behavior (no custom filename) - sequential numbers
        counter = 1
        while True:
            filename = os.path.join(output_dir, f"{counter:04d}.{extension}")
            temp_filename = filename + ".tmp"

            # Check if either the actual file or temp reservation exists
            if not os.path.exists(filename) and not os.path.exists(temp_filename):
                # Create an empty file to "reserve" this filename
                with open(temp_filename, "w") as f:
                    # Add timestamp and instance info for debugging
                    f.write(f"Reserved by process {os.getpid()} at {datetime.now().isoformat()}")

                # Add a small random delay to further reduce collision chance
                time.sleep(random.uniform(0.01, 0.05))
                return filename, temp_filename
            counter += 1

def open_folder(folder_path):
    """Open a folder in the file manager of the computer the app runs on."""
    folder_path = os.path.abspath(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    try:
        if platform.system() == "Windows":
            os.startfile(folder_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", folder_path])
        else:  # Linux or other Unix-like
            if shutil.which("xdg-open") is None:
                return f"No file manager is available on this machine. Folder: {folder_path}"
            subprocess.Popen(["xdg-open", folder_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Opened folder {folder_path}"
    except Exception as e:
        return f"Error opening folder {folder_path}: {e}"

def open_folder_notice(folder_path):
    """Open-folder buttons: report in a toast so no log box is overwritten."""
    message = open_folder(folder_path)
    if message.startswith("Opened"):
        gr.Info(message)
    else:
        gr.Warning(message)

def open_outputs_folder():
    return open_folder(DEFAULT_OUTPUT_DIR)

model_manager = ModelManager(device="cpu")

def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent, lora_path=None, lora_alpha=None):
    print(f"[CMD] Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    ensure_model_downloaded(model_choice)
    load_start = time.time()
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16
    global model_manager
    model_manager = ModelManager(device="cpu")
    folder, shards = MODEL_FILES[model_choice]
    model_dir = os.path.join("models", "Wan-AI", folder)
    t5_path = get_common_file(os.path.join("models", T5_FILE), os.path.join(model_dir, T5_FILE))
    vae_path = get_common_file(os.path.join("models", VAE_FILE), os.path.join(model_dir, VAE_FILE))
    shard_paths = [os.path.join(model_dir, shard) for shard in shards]
    if model_choice == "1.3B":
        model_manager.load_models(
            [
                shard_paths[0],
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_text":
        model_manager.load_models(
            [
                shard_paths,
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice in ("14B_image_720p", "14B_image_480p"):
        clip_path = get_common_file(os.path.join("models", CLIP_FILE), os.path.join(model_dir, CLIP_FILE))
        model_manager.load_models([clip_path], torch_dtype=torch.float32)
        model_manager.load_models(
            [
                shard_paths,
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError("Invalid model choice")
    if lora_path is not None:
        if isinstance(lora_path, list):
            for path, alpha in lora_path:
                try:
                    print(f"[CMD] Loading LoRA from {path} with alpha {alpha}")
                    if os.path.exists(path):
                        model_manager.load_lora(path, lora_alpha=alpha)
                    else:
                        print(f"[CMD] Warning: LoRA file not found: {path}")
                except Exception as e:
                    print(f"[CMD] Error loading LoRA {path}: {e}")
        else:
            try:
                print(f"[CMD] Loading LoRA from {lora_path} with alpha {lora_alpha}")
                if os.path.exists(lora_path):
                    model_manager.load_lora(lora_path, lora_alpha=lora_alpha)
                else:
                    print(f"[CMD] Warning: LoRA file not found: {lora_path}")
            except Exception as e:
                print(f"[CMD] Error loading LoRA {lora_path}: {e}")
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    try:
        num_persistent_val = int(num_persistent)
    except:
        print("[CMD] Warning: could not parse num_persistent value, defaulting to 6000000000")
        num_persistent_val = 6000000000
    print(f"num_persistent_val {num_persistent_val}")
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_val)
    print(f"[CMD] Model loaded successfully in {time.time() - load_start:.1f} seconds.")
    return pipe

def get_lora_choices():
    lora_folder = "LoRAs"
    if not os.path.exists(lora_folder):
        os.makedirs(lora_folder)
        print("[CMD] 'LoRAs' folder not found. Created 'LoRAs' folder. Please add your LoRA .safetensors files.")
    files = [f for f in os.listdir(lora_folder) if f.endswith(".safetensors")]
    choices = ["None"] + sorted(files)
    return choices

def refresh_lora_list(*current_values):
    choices = get_lora_choices()
    updates = [gr.update(choices=choices, value=value if value in choices else "None") for value in current_values]
    return tuple(updates) if updates else gr.update(choices=choices, value="None")

def apply_fast_preset():
    return 20, True, 0.15, 5.6

def cleanup_tmp_files(directory=None):
    """Clean up any remaining temporary .tmp files in the specified directory or the default outputs folder"""
    if directory is None:
        directory = DEFAULT_OUTPUT_DIR

    if not os.path.exists(directory):
        return

    try:
        tmp_files = glob.glob(os.path.join(directory, "*.tmp"))
        for tmp_file in tmp_files:
            try:
                os.remove(tmp_file)
                print(f"[CMD] Cleaned up temporary file: {tmp_file}")
            except Exception as e:
                print(f"[CMD] Failed to remove temporary file {tmp_file}: {e}")
    except Exception as e:
        print(f"[CMD] Error cleaning up temporary files: {e}")

def clean_temp_videos():
    """Clean up temporary videos that were created during re-encoding"""
    from video_utils import clean_temp_videos as utils_clean_temp_videos
    utils_clean_temp_videos()
    # Also clean up any remaining .tmp files
    cleanup_tmp_files()

# ------------------------- Models & system information -------------------------

def _file_size_gb(paths):
    total = 0
    for path in paths:
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total / 1024**3

def model_status_html():
    rows = []
    labels = {
        "1.3B": MODEL_T2V_1_3B,
        "14B_text": MODEL_T2V_14B,
        "14B_image_720p": MODEL_I2V_720P,
        "14B_image_480p": MODEL_I2V_480P,
    }
    for key in ("1.3B", "14B_text", "14B_image_720p", "14B_image_480p"):
        folder, shards = MODEL_FILES[key]
        missing = missing_model_files(key)
        size = _file_size_gb([os.path.join(MODELS_DIR, "Wan-AI", folder, shard) for shard in shards])
        if missing:
            state = f'<span class="missing">Missing</span> ({len(missing)} file(s)) - downloader option {MODEL_DOWNLOAD_OPTION[key]}'
        else:
            state = f'<span class="ok">Ready</span> ({size:.1f} GB)'
        rows.append(f"<tr><td>{labels[key]}</td><td>models/Wan-AI/{folder}</td><td>{state}</td></tr>")
    qwen_ready = os.path.isfile(os.path.join(QWEN_LOCAL_DIR, "config.json"))
    rows.append(
        "<tr><td>Prompt Enhance (Qwen2.5-14B-Instruct)</td><td>models/Qwen2.5-14B-Instruct</td><td>"
        + ('<span class="ok">Ready</span>' if qwen_ready else '<span class="missing">Not downloaded</span> - downloader option 5 (or it downloads to the Hugging Face cache on first use)')
        + "</td></tr>"
    )
    rife_ready = os.path.isfile(os.path.join("Practical-RIFE", "train_log", "flownet.pkl"))
    rows.append(
        "<tr><td>Practical-RIFE (2x / 4x FPS)</td><td>Practical-RIFE/train_log</td><td>"
        + ('<span class="ok">Ready</span>' if rife_ready else '<span class="missing">Missing</span> - run the installer again')
        + "</td></tr>"
    )
    loras = [f for f in get_lora_choices() if f != "None"]
    rows.append(f"<tr><td>LoRAs</td><td>LoRAs</td><td>{len(loras)} file(s){': ' + ', '.join(loras[:6]) if loras else ''}{' ...' if len(loras) > 6 else ''}</td></tr>")
    return (
        '<div class="model-status-table"><table><thead><tr><th>Model</th><th>Folder</th><th>Status</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )

def system_info_markdown():
    lines = []
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                line = (
                    f"- **GPU {index}:** {properties.name} (compute {properties.major}.{properties.minor}), "
                    f"{properties.total_memory / 1024**3:.1f} GB VRAM"
                )
                if index == 0:
                    # The app runs on GPU 0; asking another GPU would create a CUDA context on it
                    free, _ = torch.cuda.mem_get_info(0)
                    line += f", {free / 1024**3:.1f} GB free" + (" (the app uses this GPU)" if torch.cuda.device_count() > 1 else "")
                lines.append(line)
        else:
            lines.append("- **GPU:** CUDA is not available")
    except Exception as e:
        lines.append(f"- **GPU:** could not be read ({e})")
    memory = psutil.virtual_memory()
    lines.append(f"- **RAM:** {memory.total / 1024**3:.1f} GB, {memory.available / 1024**3:.1f} GB available")
    available = wan_video_dit.available_attention_backends()
    lines.append("- **Attention:** " + ", ".join(f"{ATTENTION_NAMES[name]} {'available' if ok else 'not available'}" for name, ok in available.items()))
    lines.append(f"- **Versions:** Python {platform.python_version()}, torch {torch.__version__} (CUDA {torch.version.cuda}), Gradio {gr.__version__}")
    lines.append(f"- **Default VRAM preset for this GPU:** {DEFAULT_VRAM_PRESET}")
    return "\n".join(lines)

def refresh_models_tab():
    return model_status_html(), system_info_markdown()

VRAM_USAGE_MARKDOWN = """
All tests were made on a secondary GPU, so they are accurate. Make sure you have that much free VRAM before you start the app. The table lists the VRAM used for each *Number of Persistent Parameters In Dit* value (81 frames).

| Model and resolution | Precision | Persistent parameters: VRAM |
|---|---|---|
| 14B Text-to-Video 1280x720 | BF16 | 0: 14.2 GB · 4.25B: 22.0 GB (24 GB GPUs) · 8.25B: 29.9 GB (32 GB GPUs) · 22B: 41.7 GB (48 GB GPUs) |
| 14B Text-to-Video 1280x720 | FP8 | 0: 14.2 GB · 8.75B: 22.0 GB (24 GB GPUs) · 22B: 27.3 GB (32 GB GPUs) |
| 14B Image-to-Video 480P 480x832 | BF16 | 0: 8.6 GB · 1.5B: 10.1 GB (12 GB) · 3.5B: 14.0 GB (16 GB) · 8B: 22.1 GB (24 GB) · 12B: 30.1 GB (32 GB) · 22B: 38.3 GB (48 GB) |
| 14B Image-to-Video 480P 480x832 | FP8 | 0: 8.6 GB · 2.5B: 9.8 GB (12 GB) · 7.5B: 14.9 GB (16 GB) · 15B: 21.7 GB (24 GB) · 22B: 23.1 GB (32 GB) |
| 14B Image-to-Video 720P 720x1280 | BF16 | 0: 18.3 GB · 3B: 22.0 GB (24 GB) · 6.75B: 29.8 GB (32 GB) · 16B: 47.0 GB (48 GB) · 22B: 47.8 GB (80 GB) |
| 14B Image-to-Video 720P 720x1280 | FP8 | 0: 18.3 GB · 6B: 22.0 GB (24 GB) · 15B: 30.8 GB (32 GB) · 22B: 32.0 GB (48 GB) |

Sage Attention needs about 3 GB less VRAM than Flash Attention at 720p, so these values leave extra headroom with it.
"""

# --- Utility Functions (No changes needed) ---
def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1: drives.append(f"{letter}:\\")
            bitmask >>= 1
        available_paths = drives
    elif platform.system() == "Darwin":
         available_paths = ["/", "/Volumes"]
    else:
        available_paths = ["/", "/mnt", "/media"]
    existing_paths = [p for p in available_paths if os.path.exists(p)]
    print(f"Allowed Gradio paths: {existing_paths}")
    return existing_paths

# Helper function for natural sorting (like Windows Explorer)
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
    return [tryint(c) for c in re.split('([0-9]+)', s)]

# ------------------------- User Interface -------------------------

def build_ui():
    cfg = config_loaded
    lora_choices = get_lora_choices()

    def lora_value(key):
        return cfg[key] if cfg[key] in lora_choices else "None"

    with gr.Blocks(title=f"SECourses Wan 2.1 {APP_VERSION}", **({} if GRADIO_6 else PAGE_STYLE)) as demo:
        with gr.Row(elem_classes=["app-header"]):
            gr.Markdown(
                f"# {APP_TITLE}\n"
                f"Image-to-Video, Video-to-Video, Text-to-Video and video extension with Wan 2.1 | "
                f"[Tutorial]({TUTORIAL_URL}) | [Premium release, updates and support]({PATREON_URL})",
                container=False,
            )
            with gr.Row(elem_classes=["header-actions"], scale=0):
                sections_button = gr.Button("⇕  Open / close all sections", elem_classes=btn("slate"), scale=0, min_width=230)
                theme_button = gr.Button("🌗  Light / dark theme", elem_classes=btn("gray"), scale=0, min_width=200)
        # Both switches are pure client-side DOM work, so they stay instant while a generation holds the queue
        sections_button.click(None, None, None, js=TOGGLE_SECTIONS_JS, queue=False, show_progress="hidden")
        theme_button.click(None, None, None, js=TOGGLE_THEME_JS, queue=False, show_progress="hidden")

        with gr.Row(elem_classes=["preset-bar"]):
            config_dropdown = gr.Dropdown(label="Config (all settings of every tab)", choices=get_config_list(), value=last_config if last_config in get_config_list() else None, scale=3)
            config_name_textbox = gr.Textbox(label="Config Name (for saving)", placeholder="Enter config name", value="", scale=3)
            save_config_button = gr.Button("💾  Save Config", elem_classes=btn("blue"), scale=1, min_width=150)
            load_config_button = gr.Button("📥  Load Config", elem_classes=btn("cyan"), scale=1, min_width=150)
            reset_config_button = gr.Button("↺  Defaults", elem_classes=btn("gold"), scale=1, min_width=130)
        config_status = gr.Markdown(f"Loaded config: **{last_config}**", elem_classes=["preset-status"])

        with gr.Tabs(elem_id="main-tabs"):
            # ------------------------------------------------------------------ Generate
            with gr.Tab("🎬 Video Generation", id="generate"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            generate_button = gr.Button("🎬  Generate", variant="primary", elem_classes=btn("emerald", "ax-lg"), scale=3)
                            cancel_button = gr.Button("⛔  Cancel", variant="stop", elem_classes=btn("red", "ax-lg"), scale=1)
                        with gr.Row():
                            fast_preset_button = gr.Button("⚡  Apply Fast Preset", elem_classes=btn("amber"))
                            enhance_button = gr.Button("✨  Prompt Enhance", elem_classes=btn("violet"))
                        prompt_box = gr.Textbox(label="Prompt (A <random: green , yellow , etc > car) will take random word with trim like : A yellow car", placeholder="Describe the video you want to generate", lines=5, value=cfg["prompt"])
                        negative_prompt = gr.Textbox(label="Negative Prompt", value=cfg["negative_prompt"], placeholder="Enter negative prompt", lines=2)

                        with gr.Accordion("🎞️ Model & Resolution", open=True):
                            with gr.Row():
                                model_choice_radio = gr.Radio(choices=MODEL_CHOICES, label="Model Choice", value=cfg["model_choice"])
                                vram_preset_radio = gr.Radio(choices=VRAM_PRESETS, label="GPU VRAM Preset", value=cfg["vram_preset"])
                            aspect_ratio_radio = gr.Radio(choices=aspect_choices_for_model(cfg["model_choice"]), label="Aspect Ratio", value=cfg["aspect_ratio"])
                            with gr.Row():
                                width_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=cfg["width"], label="Width")
                                height_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=cfg["height"], label="Height")
                            with gr.Row():
                                auto_crop_checkbox = gr.Checkbox(label="Auto Crop", value=cfg["auto_crop"])
                                auto_scale_checkbox = gr.Checkbox(label="Auto Scale", value=cfg["auto_scale"])
                                tiled_checkbox = gr.Checkbox(label="Tiled VAE Decode (Disable for 1.3B model for 12GB or more GPUs)", value=cfg["tiled"])

                        with gr.Accordion("🎛️ Generation Settings", open=True):
                            with gr.Row():
                                num_generations = gr.Number(label="Number of Generations (e.g. Generate 3 videos)", value=cfg["num_generations"], precision=0, minimum=1)
                                inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=cfg["inference_steps"], label="Inference Steps")
                                quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=cfg["quality"], label="Quality")
                            with gr.Row():
                                num_frames_slider = gr.Slider(minimum=1, maximum=300, step=1, value=cfg["num_frames"], label="Number of Frames (Always 4x+1 e.g. 17 frames = 1 second). More frames uses more VRAM and slower")
                                fps_slider = gr.Slider(minimum=8, maximum=30, step=1, value=cfg["fps"], label="FPS (for saving video - you can save as 8 FPS and 4x RIFE to get 2x duration)")
                            with gr.Row():
                                cfg_scale_slider = gr.Slider(minimum=1, maximum=12, step=0.1, value=cfg["cfg_scale"], label="CFG Scale")
                                sigma_shift_slider = gr.Slider(minimum=1, maximum=12, step=0.1, value=cfg["sigma_shift"], label="Sigma Shift")
                            with gr.Row():
                                use_random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=cfg["use_random_seed"])
                                seed_input = gr.Textbox(label="Seed (if not using random)", placeholder="Enter seed", value=cfg["seed"])
                            with gr.Row():
                                save_prompt_checkbox = gr.Checkbox(label="Save prompt to file", value=cfg["save_prompt"])
                                multiline_checkbox = gr.Checkbox(label="Multi-line prompt (each line is separate)", value=cfg["multiline"])

                        with gr.Accordion("🖼️ Input Image / Video", open=True):
                            gr.Markdown("Use the left panel to upload an image for Image-to-Video, the right panel to upload a video for Video-to-Video (1.3B model) or to extend an existing video (uses its last frame, for the Image-to-Video models).", elem_classes=["section-note"])
                            denoising_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=cfg["denoising_strength"], label="Denoising Strength (only for video-to-video)")
                            with gr.Row():
                                image_input = gr.Image(type="pil", label="Input Image (for image-to-video)", height=512)
                                video_input = gr.Video(label="Input Video (for Video-to-Video, only for 1.3B) or Extending Existing Video (Uses Last Frame, for Image-to-Video models)", format="mp4", height=512)

                        with gr.Accordion("🚀 Practical-RIFE: Increase Video FPS", open=True):
                            with gr.Row():
                                pr_rife_checkbox = gr.Checkbox(label="Apply Practical-RIFE", value=cfg["pr_rife"])
                                pr_rife_radio = gr.Radio(choices=RIFE_CHOICES, label="FPS Multiplier", value=cfg["pr_rife_multiplier"])

                        with gr.Accordion("🧠 GPU, VRAM & Speed", open=True):
                            attention_radio = gr.Radio(
                                choices=list(ATTENTION_CHOICES),
                                value=cfg["attention_backend"],
                                label="Attention",
                                info="Sage Attention makes generation up to about 2x faster on RTX 30/40/50 GPUs with the same quality and uses less VRAM. Flash Attention gives exactly the same videos as V72 for the same seed.",
                            )
                            gr.Markdown("If you get an out of VRAM error or it uses shared VRAM, reduce the Number of Persistent Parameters. FP8 may generate broken colors at the moment in I2V.", elem_classes=["section-note"])
                            with gr.Row():
                                num_persistent_text = gr.Textbox(label="Number of Persistent Parameters In Dit (VRAM)", value=cfg["num_persistent"])
                                torch_dtype_radio = gr.Radio(
                                    choices=DTYPE_CHOICES,
                                    label="torch.float8_e4m3fn is FP8 and reduces VRAM and RAM usage a lot with little quality loss. torch.bfloat16 is BF16 (max quality)",
                                    value=cfg["torch_dtype"]
                                )
                            clear_cache_checkbox = gr.Checkbox(label="Clear model from RAM and VRAM after generation (reloads the model for every video)", value=cfg["clear_cache_after_gen"])

                        with gr.Accordion("🍵 TeaCache: speeds up generation as it progresses", open=False):
                            gr.Markdown("Too big a value reduces quality and causes distortions.", elem_classes=["section-note"])
                            enable_teacache_checkbox = gr.Checkbox(label="Enable TeaCache (0.05 Threshold for 1.3b model and 0.15 for 14b models recommended)", value=cfg["enable_teacache"])
                            with gr.Row():
                                tea_cache_l1_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=cfg["tea_cache_l1_thresh"], label="Tea Cache L1 Threshold")
                                tea_cache_model_id_textbox = gr.Textbox(label="Tea Cache Model ID", value=cfg["tea_cache_model_id"], placeholder="Enter Tea Cache Model ID")

                        with gr.Accordion("🧩 LoRAs", open=False):
                            with gr.Row(elem_classes=["input-action-row"]):
                                lora_dropdown = gr.Dropdown(label="LoRA Model (Place .safetensors files in 'LoRAs' folder)", choices=lora_choices, value=lora_value("lora_model"), scale=3)
                                lora_alpha_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=cfg["lora_alpha"], label="LoRA Scale", scale=2)
                                refresh_lora_button = gr.Button("🔄  Refresh LoRAs", elem_classes=btn("teal"), scale=1, min_width=160)
                            with gr.Row():
                                show_more_lora_button = gr.Button("🧩  Show More LoRAs", elem_classes=btn("coral"))
                                more_lora_state = gr.State(False)
                            with gr.Column(visible=False) as more_lora_container:
                                with gr.Row():
                                    lora_dropdown_2 = gr.Dropdown(label="LoRA Model 2", choices=lora_choices, value=lora_value("lora_model_2"))
                                    lora_alpha_slider_2 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=cfg["lora_alpha_2"], label="LoRA Scale 2")
                                with gr.Row():
                                    lora_dropdown_3 = gr.Dropdown(label="LoRA Model 3", choices=lora_choices, value=lora_value("lora_model_3"))
                                    lora_alpha_slider_3 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=cfg["lora_alpha_3"], label="LoRA Scale 3")
                                with gr.Row():
                                    lora_dropdown_4 = gr.Dropdown(label="LoRA Model 4", choices=lora_choices, value=lora_value("lora_model_4"))
                                    lora_alpha_slider_4 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=cfg["lora_alpha_4"], label="LoRA Scale 4")

                        with gr.Accordion("🔁 Extend Video", open=False):
                            with gr.Row():
                                extend_slider = gr.Slider(minimum=1, maximum=10, step=1, value=cfg["extend_factor"], label="Extend Video Factor (1× = No Extension)")
                                extension_info_button = gr.Button("ℹ️  Extension Feature Info", elem_classes=btn("sky"), scale=0, min_width=220)
                            extension_info_output = gr.Markdown("")

                        with gr.Accordion("✨ Prompt Enhance", open=False):
                            gr.Markdown("Prompt Enhance rewrites your prompt with Qwen2.5-14B-Instruct (4-bit, about 10 GB VRAM). The model is unloaded automatically when a video generation starts.", elem_classes=["section-note"])
                            tar_lang = gr.Radio(choices=TARGET_LANGUAGES, label="Target language for prompt enhance", value=cfg["tar_lang"])

                    with gr.Column(scale=3):
                        video_output = gr.Video(label="Generated Video", height=720)
                        open_outputs_button = gr.Button("📂  Open Outputs Folder", elem_classes=btn("indigo"))
                        status_output = gr.Textbox(label="Status Log", lines=20, max_lines=40, elem_classes=["log-box"])
                        last_seed_output = gr.Textbox(label="Last Used Seed", interactive=False)

            # ------------------------------------------------------------------ Batch
            with gr.Tab("📂 Batch Processing", id="batch"):
                gr.Markdown(
                    "Batch processing generates a video for every image (.jpg, .jpeg, .png, .webp) and video (.mp4) in the input folder, "
                    "with **all the settings of the Video Generation tab** (model, resolution, steps, LoRAs, RIFE, extension...). "
                    "A text file with the same name as the image (for example `cat.txt` for `cat.png`) is used as its prompt; "
                    "otherwise the prompt of the Video Generation tab is used. Outputs are named after the input files.",
                    elem_classes=["section-note"],
                )
                with gr.Row():
                    batch_folder_input = gr.Textbox(label="Input Folder for Batch Processing", placeholder="Enter input folder path", value=cfg["batch_folder"])
                    batch_output_folder_input = gr.Textbox(label="Batch Processing Outputs Folder", placeholder="Enter batch outputs folder path", value=cfg["batch_output_folder"])
                with gr.Row():
                    skip_overwrite_checkbox = gr.Checkbox(label="Skip files whose output already exists in the outputs folder", value=cfg["skip_overwrite"])
                    save_prompt_batch_checkbox = gr.Checkbox(label="Save prompt to file (Batch)", value=cfg["save_prompt_batch"])
                with gr.Row():
                    batch_process_button = gr.Button("🎞️  Batch Process", variant="primary", elem_classes=btn("green", "ax-lg"), scale=3)
                    cancel_batch_process_button = gr.Button("⏹️  Cancel Batch Process", variant="stop", elem_classes=btn("orange", "ax-lg"), scale=1)
                    open_batch_outputs_button = gr.Button("📁  Open Batch Outputs Folder", elem_classes=btn("purple", "ax-lg"), scale=1)
                batch_status_output = gr.Textbox(label="Batch Process Status Log", lines=24, max_lines=60, elem_classes=["log-box"])

            # ------------------------------------------------------------------ Models
            with gr.Tab("📦 Models & System", id="models"):
                with gr.Row():
                    refresh_models_button = gr.Button("🔄  Refresh Status", elem_classes=btn("lime"))
                    open_models_button = gr.Button("🗂️  Open Models Folder", elem_classes=btn("fuchsia"))
                    open_loras_button = gr.Button("🧩  Open LoRAs Folder", elem_classes=btn("pink"))
                gr.Markdown("Download or verify models with **Windows_Download_Models.bat** (Windows) or **Download_Ubuntu.sh** (Linux) in the installer folder. Downloads resume after an interruption and every file is SHA256 verified.", elem_classes=["section-note"])
                model_status = gr.HTML(model_status_html())
                system_info = gr.Markdown(system_info_markdown())
                with gr.Accordion("📊 GPU VRAM usage by Number of Persistent Parameters", open=False):
                    gr.Markdown(VRAM_USAGE_MARKDOWN, elem_classes=["vram-table"])

        # ---------------------------------------------------------------------- Events
        config_components = {
            "model_choice": model_choice_radio, "vram_preset": vram_preset_radio, "aspect_ratio": aspect_ratio_radio,
            "width": width_slider, "height": height_slider, "auto_crop": auto_crop_checkbox, "auto_scale": auto_scale_checkbox,
            "tiled": tiled_checkbox, "inference_steps": inference_steps_slider, "pr_rife": pr_rife_checkbox,
            "pr_rife_multiplier": pr_rife_radio, "cfg_scale": cfg_scale_slider, "sigma_shift": sigma_shift_slider,
            "num_persistent": num_persistent_text, "torch_dtype": torch_dtype_radio,
            "lora_model": lora_dropdown, "lora_alpha": lora_alpha_slider, "lora_model_2": lora_dropdown_2,
            "lora_alpha_2": lora_alpha_slider_2, "lora_model_3": lora_dropdown_3, "lora_alpha_3": lora_alpha_slider_3,
            "lora_model_4": lora_dropdown_4, "lora_alpha_4": lora_alpha_slider_4, "clear_cache_after_gen": clear_cache_checkbox,
            "negative_prompt": negative_prompt, "save_prompt": save_prompt_checkbox, "multiline": multiline_checkbox,
            "num_generations": num_generations, "use_random_seed": use_random_seed_checkbox, "seed": seed_input,
            "quality": quality_slider, "fps": fps_slider, "num_frames": num_frames_slider, "denoising_strength": denoising_slider,
            "tar_lang": tar_lang, "batch_folder": batch_folder_input, "batch_output_folder": batch_output_folder_input,
            "skip_overwrite": skip_overwrite_checkbox, "save_prompt_batch": save_prompt_batch_checkbox,
            "enable_teacache": enable_teacache_checkbox, "tea_cache_l1_thresh": tea_cache_l1_thresh_slider,
            "tea_cache_model_id": tea_cache_model_id_textbox, "extend_factor": extend_slider,
            "attention_backend": attention_radio, "prompt": prompt_box,
        }
        config_component_list = [config_components[key] for key in CONFIG_KEYS]

        # .input fires only for user changes: a loaded config keeps its own resolution and VRAM values
        for trigger in (model_choice_radio, vram_preset_radio, torch_dtype_radio):
            trigger.input(
                fn=update_model_settings,
                inputs=[model_choice_radio, vram_preset_radio, torch_dtype_radio],
                outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
            )
        aspect_ratio_radio.input(
            fn=update_width_height,
            inputs=[aspect_ratio_radio, model_choice_radio],
            outputs=[width_slider, height_slider]
        )
        model_choice_radio.input(
            fn=update_tea_cache_model_id,
            inputs=[model_choice_radio],
            outputs=[tea_cache_model_id_textbox]
        )
        # The prompt LLM also needs the GPU, so it waits for a running generation instead of competing for VRAM
        enhance_button.click(fn=prompt_enc, inputs=[prompt_box, tar_lang], outputs=prompt_box, concurrency_id="gpu")
        generate_inputs = [
            prompt_box, tar_lang, negative_prompt, image_input, video_input, denoising_slider,
            num_generations, save_prompt_checkbox, multiline_checkbox, use_random_seed_checkbox, seed_input,
            quality_slider, fps_slider,
            model_choice_radio, vram_preset_radio, num_persistent_text, torch_dtype_radio,
            num_frames_slider,
            aspect_ratio_radio, width_slider, height_slider, auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox,
            inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
            enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
            lora_dropdown, lora_alpha_slider,
            lora_dropdown_2, lora_alpha_slider_2,
            lora_dropdown_3, lora_alpha_slider_3,
            lora_dropdown_4, lora_alpha_slider_4,
            clear_cache_checkbox,
            extend_slider,
            attention_radio,
        ]
        generate_button.click(
            fn=generate_videos_entry,
            inputs=generate_inputs,
            outputs=[video_output, status_output, last_seed_output],
            concurrency_id="gpu",
        )
        cancel_button.click(fn=cancel_generation, outputs=status_output, queue=False)
        fast_preset_button.click(fn=apply_fast_preset, inputs=[], outputs=[inference_steps_slider, enable_teacache_checkbox, tea_cache_l1_thresh_slider, sigma_shift_slider])
        open_outputs_button.click(fn=lambda: open_folder_notice(DEFAULT_OUTPUT_DIR), queue=False)
        batch_inputs = [
            prompt_box,
            batch_folder_input,
            batch_output_folder_input,
            skip_overwrite_checkbox,
            tar_lang,
            negative_prompt,
            denoising_slider,
            use_random_seed_checkbox,
            seed_input,
            quality_slider,
            fps_slider,
            model_choice_radio,
            vram_preset_radio,
            num_persistent_text,
            torch_dtype_radio,
            num_frames_slider,
            inference_steps_slider,
            aspect_ratio_radio,
            width_slider,
            height_slider,
            auto_crop_checkbox,
            auto_scale_checkbox,
            tiled_checkbox,
            cfg_scale_slider,
            sigma_shift_slider,
            save_prompt_batch_checkbox,
            pr_rife_checkbox,
            pr_rife_radio,
            lora_dropdown, lora_alpha_slider,
            lora_dropdown_2, lora_alpha_slider_2,
            lora_dropdown_3, lora_alpha_slider_3,
            lora_dropdown_4, lora_alpha_slider_4,
            enable_teacache_checkbox,
            tea_cache_l1_thresh_slider,
            tea_cache_model_id_textbox,
            clear_cache_checkbox,
            extend_slider,
            num_generations,
            attention_radio,
        ]
        batch_process_button.click(
            fn=batch_process_videos,
            inputs=batch_inputs,
            outputs=batch_status_output,
            concurrency_id="gpu",
        )
        cancel_batch_process_button.click(fn=cancel_batch_process, outputs=[status_output], queue=False)
        open_batch_outputs_button.click(fn=lambda folder: open_folder_notice((folder or "batch_outputs").strip().strip('"')), inputs=[batch_output_folder_input], queue=False)
        load_config_button.click(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[config_status] + config_component_list,
        )
        save_config_button.click(
            fn=save_config,
            inputs=[config_name_textbox] + config_component_list,
            outputs=[config_status, config_dropdown] + config_component_list,
        )
        reset_config_button.click(fn=reset_config_to_defaults, outputs=[config_status] + config_component_list)
        image_input.change(
            fn=update_target_dimensions,
            inputs=[image_input, auto_scale_checkbox, width_slider, height_slider],
            outputs=[width_slider, height_slider]
        )
        auto_scale_checkbox.input(
            fn=update_target_dimensions,
            inputs=[image_input, auto_scale_checkbox, width_slider, height_slider],
            outputs=[width_slider, height_slider]
        )
        show_more_lora_button.click(fn=toggle_lora_visibility, inputs=[more_lora_state], outputs=[more_lora_container, more_lora_state, show_more_lora_button])
        extension_info_button.click(fn=show_extension_info, inputs=[], outputs=[extension_info_output])
        lora_dropdowns = [lora_dropdown, lora_dropdown_2, lora_dropdown_3, lora_dropdown_4]
        refresh_lora_button.click(fn=refresh_lora_list, inputs=lora_dropdowns, outputs=lora_dropdowns)
        refresh_models_button.click(fn=refresh_models_tab, outputs=[model_status, system_info])
        open_models_button.click(fn=lambda: open_folder_notice(MODELS_DIR), queue=False)
        open_loras_button.click(fn=lambda: open_folder_notice(LORAS_DIR), queue=False)
    return demo

def generate_videos_entry(
    prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
    aspect_ratio, width, height, auto_crop, auto_scale, tiled,
    inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
    enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
    lora_model, lora_alpha,
    lora_model_2, lora_alpha_2,
    lora_model_3, lora_alpha_slider_3,
    lora_model_4, lora_alpha_4,
    clear_cache_after_gen, extend_factor,
    attention_choice,
    progress=gr.Progress(),
):
    """The Generate button (explicit parameters so Gradio can hand over the progress tracker)."""
    return generate_videos_ui(
        prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
        save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
        model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
        aspect_ratio, width, height, auto_crop, auto_scale, tiled,
        inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
        enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
        lora_model, lora_alpha,
        lora_model_2, lora_alpha_2,
        lora_model_3, lora_alpha_slider_3,
        lora_model_4, lora_alpha_4,
        clear_cache_after_gen, extend_factor,
        attention_choice,
        progress=progress,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use (default: models/Qwen2.5-14B-Instruct if downloaded, else Qwen/Qwen2.5-14B-Instruct).")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly.")
    parser.add_argument("--outputs", type=str, default=None, help=r'Specify the default output directory (e.g., --outputs "C:\My Videos" or --outputs "/home/user/videos").')
    parser.add_argument("--server_name", type=str, default=None, help="Address to listen on, for example 0.0.0.0 to reach the app from other computers.")
    parser.add_argument("--server_port", type=int, default=None, help="Port to listen on (default 7860, or the next free port).")
    parser.add_argument("--no_browser", action="store_true", help="Do not open the app in the browser automatically.")
    args = parser.parse_args()

    # Update default output directory if provided via CLI
    if args.outputs:
        try:
            new_output_dir = os.path.abspath(args.outputs)
            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir)
                print(f"[CMD] Created specified output directory: {new_output_dir}")
            DEFAULT_OUTPUT_DIR = new_output_dir
            print(f"[CMD] Using custom default output directory: {DEFAULT_OUTPUT_DIR}")
        except Exception as e:
            print(f"[CMD] Error setting custom output directory '{args.outputs}': {e}. Using default: {DEFAULT_OUTPUT_DIR}")

    loaded_pipeline = None
    loaded_pipeline_config = {}
    cancel_flag = False
    cancel_batch_flag = False
    prompt_expander = None
    demo = build_ui()
    # One GPU job at a time: Generate and Batch Process share a queue slot
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        share=args.share,
        inbrowser=not args.no_browser,
        server_name=args.server_name,
        server_port=args.server_port,
        allowed_paths=get_available_drives(),
        **(PAGE_STYLE if GRADIO_6 else {}),
    )
