import os
import sys
import subprocess
import random
import argparse
import time
import tempfile

import torch
import gradio as gr
from PIL import Image
import cv2

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download


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
    "9:16_low": (832, 480),
    "21:9": (1472, 624),
    "9:21": (624, 1472),
    "4:5":  (864, 1072),
    "5:4":  (1072, 864),
}


def update_vram_and_resolution(model_choice, preset):
    """
    Determines the VRAM numerical mapping and returns the list of available aspect ratios as defined by the relevant dictionary.
    """
    print(model_choice)
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
        mapping = {
            "4GB": "0",
            "6GB": "500000000",
            "8GB": "1000000000",
            "10GB": "7000000000",
            "12GB": "7000000000",
            "16GB": "7000000000",
            "24GB": "7000000000",
            "32GB": "7000000000",
            "48GB": "12000000000",
            "80GB": "12000000000"
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
            "24GB": "3000000000",
            "32GB": "6500000000",
            "48GB": "22000000000",
            "80GB": "70000000000"
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
            "24GB": "0",
            "32GB": "3500000000",
            "48GB": "12000000000",
            "80GB": "70000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "1200000000",
            "24GB": "5000000000",
            "32GB": "9500000000",
            "48GB": "20000000000",
            "80GB": "70000000000"
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
            "24GB": "0",
            "32GB": "12000000000",
            "48GB": "12000000000",
            "80GB": "70000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    return mapping.get(preset, "12000000000"), resolution_choices, default_aspect


def update_model_settings(model_choice, current_vram_preset):
    """
    When the model selection changes, update both the VRAM persistent value as well as 
    the available aspect ratios and automatically set the width and height sliders.
    """
    num_persistent_val, aspect_options, default_aspect = update_vram_and_resolution(model_choice, current_vram_preset)
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(default_aspect, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(default_aspect, (1280, 720))
    return (
        gr.update(choices=aspect_options, value=default_aspect),
        default_width,
        default_height,
        num_persistent_val
    )


def update_width_height(aspect_ratio, model_choice):
    """
    When the aspect ratio is changed, update both the width and height sliders to the default values
    based on the selected model and aspect ratio.
    """
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_ratio, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(aspect_ratio, (1280, 720))
    return default_width, default_height


def update_vram_on_change(preset, model_choice):
    """
    When the VRAM preset changes, update the num_persistent text field based on the current model.
    """
    num_persistent_val, _, _ = update_vram_and_resolution(model_choice, preset)
    return num_persistent_val


#############################################
# Auto Crop Helper Functions
#############################################
def auto_crop_image(image, target_width, target_height):
    """
    Crops and downscales the image to exactly the target resolution.
    The function first crops the image centrally to match the target aspect ratio,
    then resizes it to the target dimensions.
    """
    w, h = image.size
    target_ratio = target_width / target_height
    current_ratio = w / h

    # Crop the image to the desired aspect ratio.
    if current_ratio > target_ratio:
        # Image is too wide: crop the left and right.
        new_width = int(h * target_ratio)
        left = (w - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        # Image is too tall: crop the top and bottom.
        new_height = int(w / target_ratio)
        top = (h - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, w, bottom))

    # Resize to the target resolution.
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def auto_crop_video(video_path, target_width, target_height, desired_frame_count, desired_fps=16):
    """
    Reads a video from disk, and for each frame:
      - Downscales if the frame is larger than target dimensions.
      - Performs center crop to get exactly the target resolution.
      - Processes only a number of frames equal to desired_frame_count.
    Saves to a new file (with a _cropped suffix) and returns its path.
    The output video FPS is set to desired_fps.
    The output video duration will be desired_frame_count / desired_fps seconds.
    """
    print(f"[CMD] Starting video processing for file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[CMD] Failed to open video: {video_path}")
        return video_path
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CMD] Original video FPS: {orig_fps}")
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CMD] Original video resolution: {orig_width}x{orig_height}")
    scale = min(1.0, target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    print(f"[CMD] Scaling factor: {scale}. New intermediate resolution: {new_width}x{new_height}")
    
    base, ext = os.path.splitext(video_path)
    out_path = base + "_cropped.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, desired_fps, (target_width, target_height))
    
    frame_count = 0
    while frame_count < desired_frame_count:
        ret, frame = cap.read()
        if not ret:
            print("[CMD] No more frames available from the video.")
            break
        # Downscale if needed.
        if scale < 1.0:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        if w > target_width or h > target_height:
            left = (w - target_width) // 2
            top = (h - target_height) // 2
            frame = frame[top:top+target_height, left:left+target_width]
            print(f"[CMD] Cropped frame {frame_count+1}: left={left}, top={top}")
        else:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            print(f"[CMD] Resized frame {frame_count+1} to target resolution.")
        out.write(frame)
        frame_count += 1
    cap.release()
    out.release()
    print(f"[CMD] Finished processing video. Total frames written: {frame_count}")
    print(f"[CMD] Set output FPS to {desired_fps}. Final video duration: {frame_count/desired_fps:.2f} seconds.")
    print(f"[CMD] Output video saved to: {out_path}")
    return out_path


def prompt_enc(prompt, tar_lang):
    """
    Enhances the prompt using the prompt expander model.
    Uses the in-memory prompt expander if already loaded and does not deload the WAN pipeline.
    """
    global prompt_expander, loaded_pipeline, loaded_pipeline_config, args

    # Do not clear the WAN pipeline here.
    if prompt_expander is None:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl=False, device=0)
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    result = prompt if not prompt_output.status else prompt_output.prompt

    # Keep prompt_expander in memory for subsequent prompt enhancements.
    return result


def generate_videos(
    prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
    aspect_ratio, width, height, auto_crop, tiled, inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
    lora_model, lora_alpha
):
    """
    Main generation function now using width and height sliders as final resolution values.
    Also uses the tiled option from the new checkbox and inference steps from the same row.
    Additionally, if saving prompts is enabled, the generated text file will include extra parameters.
    
    pr_rife_enabled: boolean indicating if Practical-RIFE enhancement should be applied.
    pr_rife_radio: string ("2x FPS" or "4x FPS") indicating the FPS multiplier.
    
    The new parameters 'cfg_scale' and 'sigma_shift' are added and will be passed in common_args.
    
    IMPORTANT: For image-to-video models, if auto_crop is enabled, the image is auto processed and saved
    in the folder "auto_pre_processed_images" immediately (before starting video generation) with the same
    base name as the final video output.
    
    New LoRA support:
      - If a LoRA model is selected (i.e. not "None") then the pipeline will be reloaded with that LoRA applied.
      - The LoRA scale (lora_alpha) is also tracked.
      - If LoRA is unselected (“None”), the pipeline is reloaded without a LoRA.
    """
    global loaded_pipeline, loaded_pipeline_config, cancel_flag
    cancel_flag = False  # reset cancellation flag at start
    log_text = ""
    last_used_seed = None
    last_video_path = ""
    overall_start_time = time.time()  # overall timer

    # Determine which effective model is used.
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
        return "", "Invalid model choice.", ""
    
    target_width = int(width)
    target_height = int(height)

    # Process video input if applicable.
    if model_choice == "1.3B" and input_video is not None:
        original_video_path = input_video if isinstance(input_video, str) else input_video.name
        cap = cv2.VideoCapture(original_video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            effective_num_frames = min(int(num_frames), total_frames)
            print(f"[CMD] Detected input video frame count: {total_frames}, using effective frame count: {effective_num_frames}")
        else:
            effective_num_frames = int(num_frames)
            print("[CMD] Could not open input video to determine frame count, using provided frame count")
        cap.release()
    else:
        effective_num_frames = int(num_frames)

    # Pre-process based on auto_crop and input type.
    if auto_crop:
        if model_choice == "1.3B" and input_video is not None:
            input_video_path = input_video if isinstance(input_video, str) else input_video.name
            print(f"[CMD] Auto cropping input video: {input_video_path}")
            input_video_path = auto_crop_video(input_video_path, target_width, target_height, effective_num_frames, desired_fps=16)
            input_video = input_video_path
        elif input_image is not None and model_choice in ["14B_image_720p", "14B_image_480p"]:
            # For image-to-video models, keep a copy of the original image to process per iteration.
            original_image = input_image.copy()
    else:
        if input_image is not None:
            original_image = input_image.copy()

    vram_value = num_persistent_input

    # Process LoRA input:
    if lora_model == "None" or not lora_model:
        effective_lora_model = None
    else:
        effective_lora_model = os.path.join("LoRAs", lora_model)

    # Prepare the current configuration dictionary (including LoRA settings).
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
        "lora_model": effective_lora_model,
        "lora_alpha": lora_alpha,
    }

    # If the loaded pipeline config has changed (including changes in LoRA selection/scale), reload the pipeline.
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        if effective_lora_model is not None:
            print(f"[CMD] Applying LoRA: {effective_lora_model} with scale {lora_alpha}")
        else:
            print("[CMD] No LoRA selected. Using base model.")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value,
                                              lora_path=effective_lora_model, lora_alpha=lora_alpha)
        loaded_pipeline_config = current_config

    if multi_line:
        prompts_list = [line.strip() for line in prompt.splitlines() if line.strip()]
    else:
        prompts_list = [prompt.strip()]

    total_iterations = len(prompts_list) * int(num_generations)
    iteration = 0

    for p in prompts_list:
        for i in range(int(num_generations)):
            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user.\n"
                print("[CMD] Generation cancelled by user.")
                duration = time.time() - overall_start_time
                log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                loaded_pipeline = None
                loaded_pipeline_config = {}
                return "", log_text, str(last_used_seed or "")
            iteration += 1

            iter_start = time.time()

            log_text += f"[CMD] Generating video {iteration} of {total_iterations} with prompt: {p}\n"
            print(f"[CMD] Generating video {iteration}/{total_iterations} with prompt: {p}")

            # Optionally enhance prompt.
            enhanced_prompt = p

            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                try:
                    current_seed = int(seed_input) if seed_input.strip() != "" else 0
                except Exception as e:
                    current_seed = 0
            last_used_seed = current_seed
            print(f"[CMD] Using resolution: width={target_width}  height={target_height}")

            common_args = {
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(inference_steps),
                "seed": current_seed,
                "tiled": tiled,
                "width": target_width,
                "height": target_height,
                "num_frames": effective_num_frames,
                "cfg_scale": cfg_scale,
                "sigma_shift": sigma_shift,
            }

            # Choose branch according to model type.
            if model_choice == "1.3B":
                if input_video is not None:
                    input_video_path = input_video if isinstance(input_video, str) else input_video.name
                    print(f"[CMD] Processing video-to-video with input video: {input_video_path}")
                    video_obj = VideoData(input_video_path, height=target_height, width=target_width)
                    video_data = loaded_pipeline(input_video=video_obj, denoising_strength=denoising_strength, **common_args)
                else:
                    video_data = loaded_pipeline(**common_args)
                video_filename = get_next_filename(".mp4")
            elif model_choice == "14B_text":
                video_data = loaded_pipeline(**common_args)
                video_filename = get_next_filename(".mp4")
            elif model_choice in ["14B_image_720p", "14B_image_480p"]:
                if input_image is None:
                    err_msg = "[CMD] Error: Image model selected but no image provided."
                    print(err_msg)
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    return "", err_msg, str(last_used_seed or "")
                # For image-to-video, process the image *per iteration* so that we can save the preprocessed image
                if auto_crop:
                    processed_image = auto_crop_image(original_image, target_width, target_height)
                else:
                    processed_image = original_image
                video_filename = get_next_filename(".mp4")
                # Save preprocessed image BEFORE generating the video.
                preprocessed_folder = "auto_pre_processed_images"
                if not os.path.exists(preprocessed_folder):
                    os.makedirs(preprocessed_folder)
                base_name = os.path.splitext(os.path.basename(video_filename))[0]
                preprocessed_image_filename = os.path.join(preprocessed_folder, f"{base_name}.png")
                processed_image.save(preprocessed_image_filename)
                log_text += f"[CMD] Saved auto processed image: {preprocessed_image_filename}\n"
                print(f"[CMD] Saved auto processed image: {preprocessed_image_filename}")
                video_data = loaded_pipeline(input_image=processed_image, **common_args)
            else:
                err_msg = "[CMD] Invalid combination of inputs."
                print(err_msg)
                loaded_pipeline = None
                loaded_pipeline_config = {}
                return "", err_msg, str(last_used_seed or "")

            save_video(video_data, video_filename, fps=fps, quality=quality)
            log_text += f"[CMD] Saved video: {video_filename}\n"
            print(f"[CMD] Saved video: {video_filename}")

            if save_prompt:
                text_filename = os.path.splitext(video_filename)[0] + ".txt"
                generation_details = ""
                generation_details += f"Prompt: {enhanced_prompt}\n"
                generation_details += f"Negative Prompt: {negative_prompt}\n"
                generation_details += f"Used Model: {model_choice_radio}\n"
                generation_details += f"Number of Inference Steps: {inference_steps}\n"
                generation_details += f"Seed: {current_seed}\n"
                generation_details += f"Number of Frames: {effective_num_frames}\n"
                if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)" and input_video is not None:
                    generation_details += f"Denoising Strength: {denoising_strength}\n"
                else:
                    generation_details += "Denoising Strength: N/A\n"
                # Log LoRA usage
                if lora_model and lora_model != "None":
                    generation_details += f"LoRA Model: {lora_model} with scale {lora_alpha}\n"
                else:
                    generation_details += "LoRA Model: None\n"
                # Log the precision used based on torch_dtype.
                generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
                generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
                generation_details += f"Generation Duration: {time.time()-iter_start:.2f} seconds / {(time.time()-iter_start)/60:.2f} minutes\n"
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(generation_details)
                log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
                print(f"[CMD] Saved prompt and parameters: {text_filename}")

            last_video_path = video_filename

    if pr_rife_enabled and last_video_path:
        print(f"[CMD] Applying Practical-RIFE with multiplier {pr_rife_radio} on video {last_video_path}")
        multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
        improved_video = os.path.join("outputs", "improved_" + os.path.basename(last_video_path))
        model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
        cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{model_dir}" --multi={multiplier_val} --video="{last_video_path}" --output="{improved_video}"'
        print(f"[CMD] Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"[CMD] Practical-RIFE finished. Improved video saved to: {improved_video}")
        last_video_path = improved_video
        log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"

    overall_duration = time.time() - overall_start_time
    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
    log_text += f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes). Last used seed: {last_used_seed}\n"
    print(f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds. Last used seed: {last_used_seed}")

    loaded_pipeline = None
    loaded_pipeline_config = {}

    return last_video_path, log_text, str(last_used_seed or "")


def cancel_generation():
    """
    Sets the global cancel flag so that generation loops can end early.
    """
    global cancel_flag
    cancel_flag = True
    print("[CMD] Cancel button pressed.")
    return "Cancelling generation..."


def batch_process_videos(
    default_prompt, folder_path, batch_output_folder, skip_overwrite, tar_lang, negative_prompt, denoising_strength,
    use_random_seed, seed_input, quality, fps, model_choice_radio, vram_preset, num_persistent_input,
    torch_dtype, num_frames, inference_steps, aspect_ratio, width, height, auto_crop,
    save_prompt, pr_rife_enabled, pr_rife_radio, lora_model, lora_alpha
):
    """
    Processes a folder of images for image-to-video generation in batch using width and height as resolution.
    Applies Practical-RIFE if enabled.
    Additionally, if saving prompt parameters is enabled, a text file with the generation details
    is saved alongside each generated video.

    Now, if a prompt text file is not found for an image, the default user entered prompt (from the regular prompt box)
    is used instead and a corresponding message is printed on cmd.

    LoRA settings are also logged.
    """
    global loaded_pipeline, loaded_pipeline_config, cancel_batch_flag
    cancel_batch_flag = False  # reset cancellation flag for batch process
    log_text = ""
    
    # For batch processing, we expect only the image-to-video models.
    if model_choice_radio not in ["WAN 2.1 14B Image-to-Video 720P", "WAN 2.1 14B Image-to-Video 480P"]:
        log_text += "[CMD] Batch processing currently only supports the WAN 2.1 14B Image-to-Video models.\n"
        return log_text

    target_width = int(width)
    target_height = int(height)
    
    vram_value = num_persistent_input
    if model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
    else:  # WAN 2.1 14B Image-to-Video 480P
        model_choice = "14B_image_480p"
    
    if lora_model == "None" or not lora_model:
        effective_lora_model = None
    else:
        effective_lora_model = os.path.join("LoRAs", lora_model)
    
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
        "lora_model": effective_lora_model,
        "lora_alpha": lora_alpha,
    }
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        if effective_lora_model is not None:
            print(f"[CMD] Applying LoRA in batch: {effective_lora_model} with scale {lora_alpha}")
        else:
            print("[CMD] No LoRA selected for batch. Using base model.")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_lora_model, lora_alpha=lora_alpha)
        loaded_pipeline_config = current_config

    common_args_base = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(inference_steps),
        "tiled": True,
        "width": target_width,
        "height": target_height,
        "num_frames": int(num_frames),
    }
    
    if not os.path.isdir(folder_path):
        log_text += f"[CMD] Provided folder path does not exist: {folder_path}\n"
        return log_text

    if not os.path.exists(batch_output_folder):
        os.makedirs(batch_output_folder)
        log_text += f"[CMD] Created batch processing outputs folder: {batch_output_folder}\n"

    files = os.listdir(folder_path)
    images = [f for f in files if os.path.splitext(f)[1].lower() in [".jpg", ".png", ".jpeg"]]
    total_files = len(images)
    log_text += f"[CMD] Found {total_files} image files in folder {folder_path}\n"
    
    for image_file in images:
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            print("[CMD] Batch processing cancelled by user.")
            break

        iter_start = time.time()
        base, ext = os.path.splitext(image_file)
        prompt_path = os.path.join(folder_path, base + ".txt")
        if not os.path.exists(prompt_path):
            log_text += f"[CMD] No prompt txt found for {image_file}, using user entered prompt: {default_prompt}\n"
            print(f"[CMD] No prompt txt found for {image_file}, using user entered prompt: {default_prompt}")
            prompt_content = default_prompt
        else:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
            if prompt_content == "":
                log_text += f"[CMD] Prompt txt {base + '.txt'} is empty, using user entered prompt: {default_prompt}\n"
                print(f"[CMD] Prompt txt {base + '.txt'} is empty, using user entered prompt: {default_prompt}")
                prompt_content = default_prompt
            else:
                log_text += f"[CMD] Using user made prompt txt for {image_file}: {prompt_content}\n"
                print(f"[CMD] Using user made prompt txt for {image_file}: {prompt_content}")

        output_filename = os.path.join(batch_output_folder, base + ".mp4")
        if skip_overwrite and os.path.exists(output_filename):
            log_text += f"[CMD] Output video {output_filename} already exists, skipping {image_file} due to skip overwrite.\n"
            print(f"[CMD] Output video {output_filename} already exists, skipping {image_file}.")
            continue

        if use_random_seed:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            try:
                current_seed = int(seed_input) if seed_input.strip() != "" else 0
            except Exception as e:
                current_seed = 0

        common_args = common_args_base.copy()
        common_args["prompt"] = prompt_content
        common_args["seed"] = current_seed

        log_text += f"[CMD] Processing {image_file} with prompt from {base + '.txt'} (or user entered prompt) and seed {current_seed}\n"
        print(f"[CMD] Processing {image_file} with prompt from {base + '.txt'} (or user entered prompt) and seed {current_seed}")
        
        try:
            image_path = os.path.join(folder_path, image_file)
            image_obj = Image.open(image_path).convert("RGB")
        except Exception as e:
            log_text += f"[CMD] Failed to open image {image_file}: {str(e)}\n"
            print(f"[CMD] Failed to open image {image_file}: {str(e)}")
            continue

        if auto_crop:
            processed_image = auto_crop_image(image_obj, target_width, target_height)
        else:
            processed_image = image_obj
            
        video_data = loaded_pipeline(input_image=processed_image, **common_args)
        save_video(video_data, output_filename, fps=fps, quality=quality)
        log_text += f"[CMD] Saved batch generated video: {output_filename}\n"
        print(f"[CMD] Saved batch generated video: {output_filename}")

        # Save the preprocessed image BEFORE video generation is complete.
        if auto_crop:
            preprocessed_folder = "auto_pre_processed_images"
            if not os.path.exists(preprocessed_folder):
                os.makedirs(preprocessed_folder)
            base_name = os.path.splitext(os.path.basename(output_filename))[0]
            preprocessed_image_filename = os.path.join(preprocessed_folder, f"{base_name}.png")
            processed_image.save(preprocessed_image_filename)
            log_text += f"[CMD] Saved auto processed image: {preprocessed_image_filename}\n"
            print(f"[CMD] Saved auto processed image: {preprocessed_image_filename}")
        
        generation_duration = time.time() - iter_start
        if save_prompt:
            text_filename = os.path.splitext(output_filename)[0] + ".txt"
            generation_details = ""
            generation_details += f"Prompt: {prompt_content}\n"
            generation_details += f"Negative Prompt: {negative_prompt}\n"
            generation_details += f"Used Model: {model_choice_radio}\n"
            generation_details += f"Number of Inference Steps: {inference_steps}\n"
            generation_details += f"Seed: {current_seed}\n"
            generation_details += f"Number of Frames: {num_frames}\n"
            generation_details += f"Denoising Strength: {denoising_strength}\n"
            # Log LoRA usage in batch
            if lora_model and lora_model != "None":
                generation_details += f"LoRA Model: {lora_model} with scale {lora_alpha}\n"
            else:
                generation_details += "LoRA Model: None\n"
            # Log the precision used based on torch_dtype.
            generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
            generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
            generation_details += f"Generation Duration: {generation_duration:.2f} seconds / {(generation_duration/60):.2f} minutes\n"
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(generation_details)
            log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
            print(f"[CMD] Saved prompt and parameters: {text_filename}")

        if pr_rife_enabled:
            print(f"[CMD] Applying Practical-RIFE with multiplier {pr_rife_radio} on video {output_filename}")
            multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
            improved_video = os.path.join(batch_output_folder, "improved_" + os.path.basename(output_filename))
            model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
            cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{model_dir}" --multi={multiplier_val} --video="{output_filename}" --output="{improved_video}"'
            print(f"[CMD] Running command: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            print(f"[CMD] Practical-RIFE finished. Improved video saved to: {improved_video}")
            log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"

    loaded_pipeline = None
    loaded_pipeline_config = {}

    return log_text


def cancel_batch_process():
    """
    Sets the global cancel flag so that batch processing can end early.
    """
    global cancel_batch_flag
    cancel_batch_flag = True
    print("[CMD] Batch process cancel button pressed.")
    return "Cancelling batch process..."


def get_next_filename(extension):
    """
    Returns the next available file path under the "outputs" folder.
    The filename is in the format 00001.ext, 00002.ext, etc.
    """
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    existing_files = [f for f in os.listdir(outputs_dir) if f.endswith(extension)]
    current_numbers = []
    for file in existing_files:
        try:
            num = int(os.path.splitext(file)[0])
            current_numbers.append(num)
        except Exception as e:
            continue
    next_number = max(current_numbers, default=0) + 1
    return os.path.join(outputs_dir, f"{next_number:05d}{extension}")


def open_outputs_folder():
    """
    Opens the outputs folder using the default file explorer on Windows or Linux.
    """
    outputs_dir = os.path.abspath("outputs")
    if os.name == 'nt':
        os.startfile(outputs_dir)
    elif os.name == 'posix':
        subprocess.Popen(["xdg-open", outputs_dir])
    else:
        print("[CMD] Opening folder not supported on this OS.")
    return f"Opened {outputs_dir}"


def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent, lora_path=None, lora_alpha=None):
    """
    Loads the appropriate WAN pipeline based on:
      - model_choice: one of "1.3B", "14B_text", "14B_image_720p", or "14B_image_480p"
      - torch_dtype_str: either "torch.float8_e4m3fn" or "torch.bfloat16"
      - num_persistent: VRAM related parameter (can be an integer or None)
      - lora_path: path to a LoRA .safetensors file if selected, else None.
      - lora_alpha: scaling factor for the LoRA.
    
    When a LoRA is provided, the model_manager loads the LoRA weights _before_ instantiating the pipeline.
    Then the pipeline is loaded only once with the LoRA included.
    The pipeline is then enabled to manage VRAM with the provided persistent parameter.
    """
    print(f"[CMD] Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16

    model_manager = ModelManager(device="cpu")
    if model_choice == "1.3B":
        model_manager.load_models(
            [
                "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_text":
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
                ],
                "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_image_720p":
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
                ],
                "models/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_image_480p":
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
                ],
                "models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError("Invalid model choice")
    
    if lora_path is not None:
        print(f"[CMD] Loading LoRA from {lora_path} with alpha {lora_alpha}")
        model_manager.load_lora(lora_path, lora_alpha=lora_alpha)

    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)

    if str(num_persistent).strip().lower() == "none":
        num_persistent_val = None
    else:
        try:
            num_persistent_val = int(num_persistent)
        except Exception as e:
            print("[CMD] Warning: could not parse num_persistent_param_in_dit value, defaulting to 6000000000")
            num_persistent_val = 6000000000
    print(f"num_persistent_val {num_persistent_val}")
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_val)
    print("[CMD] Model loaded successfully.")
    return pipe


def get_lora_choices():
    """
    Scans the 'LoRAs' folder for .safetensors files and returns a list of choices,
    including a "None" option as the first entry.
    """
    lora_folder = "LoRAs"
    if not os.path.exists(lora_folder):
        os.makedirs(lora_folder)
        print("[CMD] 'LoRAs' folder not found. Created 'LoRAs' folder. Please add your LoRA .safetensors files.")
    files = [f for f in os.listdir(lora_folder) if f.endswith(".safetensors")]
    choices = ["None"] + sorted(files)
    return choices


def refresh_lora_list():
    """
    Scans the 'LoRAs' folder for .safetensors files and returns an update with the list.
    """
    return gr.update(choices=get_lora_choices(), value="None")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Prompt Enhance arguments
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    # Gradio share option
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly.")
    args = parser.parse_args()

    # Global pipeline variables
    loaded_pipeline = None
    loaded_pipeline_config = {}
    cancel_flag = False
    cancel_batch_flag = False
    prompt_expander = None

    with gr.Blocks() as demo:
        gr.Markdown("SECourses Wan 2.1 I2V - V2V - T2V Advanced Gradio APP V23 | Tutorial : https://youtu.be/hnAhveNy-8s | Source : https://www.patreon.com/posts/123105403")
        with gr.Row():
            with gr.Column(scale=4):
                # Model & Resolution settings
                with gr.Row():
                    gr.Markdown("### Model & Resolution")
                with gr.Row():
                    model_choice_radio = gr.Radio(
                        choices=[
                            "WAN 2.1 1.3B (Text/Video-to-Video)",
                            "WAN 2.1 14B Text-to-Video",
                            "WAN 2.1 14B Image-to-Video 720P",
                            "WAN 2.1 14B Image-to-Video 480P"
                        ],
                        label="Model Choice",
                        value="WAN 2.1 1.3B (Text/Video-to-Video)"
                    )
                    vram_preset_radio = gr.Radio(
                        choices=["4GB", "6GB", "8GB", "10GB", "12GB", "16GB", "24GB", "32GB", "48GB", "80GB"],
                        label="GPU VRAM Preset",
                        value="24GB"
                    )
                with gr.Row():
                    # Fix: set default aspect ratios so that they are visible immediately.
                    aspect_ratio_radio = gr.Radio(
                        choices=list(ASPECT_RATIOS_1_3b.keys()),
                        label="Aspect Ratio",
                        value="16:9"
                    )
                # Combined row: width, height, auto crop, tiled and inference steps.
                with gr.Row():
                    width_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=832, label="Width")
                    height_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=480, label="Height")
                    auto_crop_checkbox = gr.Checkbox(label="Auto Crop", value=True)
                    tiled_checkbox = gr.Checkbox(label="Tiled VAE Decode (Disable for 1.3B model for 12GB or more GPUs)", value=True)
                    inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Inference Steps")
                # New Practical-RIFE options and new CFG Scale & Sigma Shift sliders.
                gr.Markdown("### Increase Video FPS with Practical-RIFE")
                with gr.Row():
                    pr_rife_checkbox = gr.Checkbox(label="Apply Practical-RIFE", value=True)
                    pr_rife_radio = gr.Radio(choices=["2x FPS", "4x FPS"], label="FPS Multiplier", value="2x FPS")
                    cfg_scale_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=6.0, label="CFG Scale")
                    sigma_shift_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=6.0, label="Sigma Shift")
                gr.Markdown("### GPU Settings")
                with gr.Row():
                    num_persistent_text = gr.Textbox(label="Number of Persistent Parameters In Dit (VRAM)", value="12000000000")
                    torch_dtype_radio = gr.Radio(
                        choices=["torch.float8_e4m3fn", "torch.bfloat16"],
                        label="Torch DType: float8 (FP8) reduces VRAM and RAM Usage",
                        value="torch.bfloat16"
                    )
                # --- New LoRA support UI elements under the prompt box --- 
                with gr.Row():
                    lora_dropdown = gr.Dropdown(
                        label="LoRA Model (Place .safetensors files in 'LoRAs' folder)",
                        choices=get_lora_choices(),  # Now populated on load
                        value="None"
                    )
                    lora_alpha_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="LoRA Scale")
                    refresh_lora_button = gr.Button("Refresh LoRAs")
                # -----------------------------------------------------------
                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    cancel_button = gr.Button("Cancel")
                # Prompt input and enhancement.
                prompt_box = gr.Textbox(label="Prompt", placeholder="Describe the video you want to generate", lines=5)
                with gr.Row():
                    tar_lang = gr.Radio(choices=["CH", "EN"], label="Target language for prompt enhance", value="EN")
                    enhance_button = gr.Button("Prompt Enhance")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="still and motionless picture, static", placeholder="Enter negative prompt", lines=2)
                with gr.Row():
                    save_prompt_checkbox = gr.Checkbox(label="Save prompt to file", value=True)
                    multiline_checkbox = gr.Checkbox(label="Multi-line prompt (each line is separate)", value=False)
                num_generations = gr.Number(label="Number of Generations", value=1, precision=0)
                with gr.Row():
                    use_random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=True)
                    seed_input = gr.Textbox(label="Seed (if not using random)", placeholder="Enter seed", value="")
                with gr.Row():
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Quality")
                    fps_slider = gr.Slider(minimum=8, maximum=30, step=1, value=16, label="FPS (for saving video)")
                    num_frames_slider = gr.Slider(minimum=1, maximum=300, step=1, value=81, label="Number of Frames")
                # Image and Video inputs.
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Input Image (for image-to-video)", height=512)
                    video_input = gr.Video(label="Input Video (for video-to-video, only for 1.3B)", format="mp4", height=512)
                denoising_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                                             label="Denoising Strength (only for video-to-video)")
            with gr.Column(scale=3):
                video_output = gr.Video(label="Generated Video", height=720)
                gr.Markdown("### Batch Image-to-Video Processing")
                batch_folder_input = gr.Textbox(label="Input Folder for Batch Processing", placeholder="Enter input folder path", value="batch_inputs")
                batch_output_folder_input = gr.Textbox(label="Batch Processing Outputs Folder", placeholder="Enter batch outputs folder path", value="batch_outputs")
                with gr.Row():
                    skip_overwrite_checkbox = gr.Checkbox(label="Skip Overwrite if Output Exists", value=True)
                    save_prompt_batch_checkbox = gr.Checkbox(label="Save prompt to file (Batch)", value=True)
                with gr.Row():
                    batch_process_button = gr.Button("Batch Process")
                    cancel_batch_process_button = gr.Button("Cancel Batch Process")
                batch_status_output = gr.Textbox(label="Batch Process Status Log", lines=10)
                status_output = gr.Textbox(label="Status Log", lines=20)
                last_seed_output = gr.Textbox(label="Last Used Seed", interactive=False)
                open_outputs_button = gr.Button("Open Outputs Folder")

        # Button events and interactions.
        enhance_button.click(fn=prompt_enc, inputs=[prompt_box, tar_lang], outputs=prompt_box)
        generate_button.click(
            fn=generate_videos,
            inputs=[
                prompt_box, tar_lang, negative_prompt, image_input, video_input, denoising_slider,
                num_generations, save_prompt_checkbox, multiline_checkbox, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider,
                model_choice_radio, vram_preset_radio, num_persistent_text, torch_dtype_radio,
                num_frames_slider,
                aspect_ratio_radio, width_slider, height_slider, auto_crop_checkbox, tiled_checkbox, inference_steps_slider,
                pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                lora_dropdown, lora_alpha_slider
            ],
            outputs=[video_output, status_output, last_seed_output]
        )
        cancel_button.click(fn=cancel_generation, outputs=status_output)
        open_outputs_button.click(fn=open_outputs_folder, outputs=status_output)
        batch_process_button.click(
            fn=batch_process_videos,
            inputs=[
                prompt_box,  # default prompt to use if txt file is missing
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
                save_prompt_batch_checkbox,
                pr_rife_checkbox, 
                pr_rife_radio,
                lora_dropdown,
                lora_alpha_slider
            ],
            outputs=batch_status_output
        )
        cancel_batch_process_button.click(fn=cancel_batch_process, outputs=batch_status_output)
        model_choice_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        aspect_ratio_radio.change(
            fn=update_width_height,
            inputs=[aspect_ratio_radio, model_choice_radio],
            outputs=[width_slider, height_slider]
        )
        vram_preset_radio.change(
            fn=update_vram_on_change,
            inputs=[vram_preset_radio, model_choice_radio],
            outputs=num_persistent_text
        )
        refresh_lora_button.click(fn=refresh_lora_list, inputs=[], outputs=lora_dropdown)

        demo.launch(share=args.share, inbrowser=True)