import os
import cv2
import re
import subprocess
import time
import numpy as np
from pathlib import Path
import traceback

def reencode_video_to_16fps(input_video_path, num_frames, target_width=None, target_height=None):
    """
    Re-encodes the input video to 16 FPS and trims it to match the desired frame count.
    Also handles resizing to match target dimensions if provided.
    
    Args:
        input_video_path: Path to the input video
        num_frames: Number of frames requested by the user
        target_width: Target width for the output video (optional)
        target_height: Target height for the output video (optional)
        
    Returns:
        Path to the re-encoded video
    """
    cap = None
    verify_cap = None
    frames_dir = None
    temp_audio_file = None
    temp_video_file = None
    timestamp = int(time.time())
    
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"[CMD] Could not open video {input_video_path}")
            return input_video_path
        
        # Get input video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # If target dimensions aren't specified, use input dimensions
        if target_width is None:
            target_width = input_width
        if target_height is None:
            target_height = input_height
            
        print(f"[CMD] Input video: {total_frames} frames at {input_fps} FPS, dimensions: {input_width}x{input_height}")
        print(f"[CMD] Target: {num_frames} frames at 16 FPS, dimensions: {target_width}x{target_height}")
        
        # Determine if reencoding is needed
        needs_reencoding = False
        
        if abs(input_fps - 16) > 0.01:
            print(f"[CMD] Input video needs re-encoding: FPS is {input_fps} instead of 16")
            needs_reencoding = True
        
        if total_frames > (num_frames + 5) or total_frames < num_frames:
            print(f"[CMD] Input video needs re-encoding: frame count mismatch ({total_frames} vs {num_frames})")
            needs_reencoding = True
        
        if input_width != target_width or input_height != target_height:
            print(f"[CMD] Input video needs re-encoding: dimension mismatch ({input_width}x{input_height} vs {target_width}x{target_height})")
            needs_reencoding = True
        
        if not needs_reencoding:
            cap.release()
            print(f"[CMD] Video already meets requirements, no re-encoding needed")
            return input_video_path
        
        # Create output directory
        output_folder = "auto_pre_processed_videos"
        os.makedirs(output_folder, exist_ok=True)
        
        # Sanitize filename - only allow English letters and underscore, max 75 chars
        input_name = Path(input_video_path).name
        base_name = os.path.splitext(input_name)[0]
        ext = os.path.splitext(input_name)[1]
        
        # Sanitize the base name - replace non-alphanumeric chars with underscore
        sanitized_name = ""
        for char in base_name:
            if char.isalnum() and char.isascii():
                sanitized_name += char
            else:
                sanitized_name += "_"
        
        # Truncate if longer than 60 chars (75 - "reencoded_" - timestamp - extension)
        if len(sanitized_name) > 60:
            sanitized_name = sanitized_name[:60]
        
        reencoded_video = os.path.join(output_folder, f"reencoded_{timestamp}_{sanitized_name}{ext}")
        frames_dir = os.path.join(output_folder, f"temp_frames_{timestamp}")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Calculate duration to extract based on num_frames at 16fps
        target_duration_sec = (num_frames - 1) / 16
        input_duration_sec = total_frames / input_fps
        
        # Use the shorter of the two durations to ensure we don't exceed input length
        extract_duration_sec = min(target_duration_sec, input_duration_sec)
        print(f"[CMD] Extracting {extract_duration_sec:.2f} seconds from input video")
        
        # Step 1: Extract exactly num_frames frames, with proper resizing and aspect ratio
        frames = []
        
        # Calculate input aspect ratio
        input_aspect = input_width / input_height
        target_aspect = target_width / target_height
        
        # Determine how many frames to extract from input
        frames_to_extract = min(num_frames, total_frames)
        
        # For any missing frames, we'll duplicate the first frame
        missing_frames = max(0, num_frames - total_frames)
        if missing_frames > 0:
            print(f"[CMD] Input video has fewer frames than needed. Will duplicate first frame {missing_frames} times.")
        
        # Read the first frame (which might need to be duplicated)
        success, first_frame = cap.read()
        if not success:
            cap.release()
            print(f"[CMD] Could not read first frame from video")
            return input_video_path
            
        # Process the first frame (resize/crop)
        if abs(input_aspect - target_aspect) < 0.01:
            # Simple resize if aspect ratios are close enough
            first_frame_processed = cv2.resize(first_frame, (target_width, target_height))
        else:
            # Need to crop and scale to maintain aspect ratio
            if input_aspect > target_aspect:
                # Input is wider than target - crop width
                new_width = int(input_height * target_aspect)
                crop_x = int((input_width - new_width) / 2)
                cropped = first_frame[:, crop_x:crop_x+new_width]
                first_frame_processed = cv2.resize(cropped, (target_width, target_height))
            else:
                # Input is taller than target - crop height
                new_height = int(input_width / target_aspect)
                crop_y = int((input_height - new_height) / 2)
                cropped = first_frame[crop_y:crop_y+new_height, :]
                first_frame_processed = cv2.resize(cropped, (target_width, target_height))
        
        # Add duplicated first frames if needed
        for i in range(missing_frames):
            frames.append(first_frame_processed.copy())
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, first_frame_processed)
            
        # Process the remaining frames
        frame_count = missing_frames
        frames.append(first_frame_processed)  # Add the actual first frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_path, first_frame_processed)
        frame_count += 1
        
        # If we need more frames, continue reading from input
        if frame_count < num_frames:
            # Calculate frame step to evenly distribute frames if input has more frames than needed
            if total_frames > 1 and frames_to_extract > 1:
                frame_step = (total_frames - 1) / (frames_to_extract - 1)
            else:
                frame_step = 1
                
            current_pos = 0
            
            while frame_count < num_frames and current_pos < total_frames - 1:
                # Calculate next frame to extract
                next_pos = int(min(current_pos + frame_step, total_frames - 1))
                if next_pos <= current_pos:
                    next_pos = current_pos + 1
                
                # Set position
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame (resize/crop)
                if abs(input_aspect - target_aspect) < 0.01:
                    # Simple resize
                    frame_processed = cv2.resize(frame, (target_width, target_height))
                else:
                    # Crop and scale
                    if input_aspect > target_aspect:
                        # Input is wider than target - crop width
                        new_width = int(input_height * target_aspect)
                        crop_x = int((input_width - new_width) / 2)
                        cropped = frame[:, crop_x:crop_x+new_width]
                        frame_processed = cv2.resize(cropped, (target_width, target_height))
                    else:
                        # Input is taller than target - crop height
                        new_height = int(input_width / target_aspect)
                        crop_y = int((input_height - new_height) / 2)
                        cropped = frame[crop_y:crop_y+new_height, :]
                        frame_processed = cv2.resize(cropped, (target_width, target_height))
                
                frames.append(frame_processed)
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_path, frame_processed)
                
                # Update positions
                current_pos = next_pos
                frame_count += 1
        
        cap.release()
        cap = None
        
        # Ensure we have exactly num_frames
        if len(frames) != num_frames:
            print(f"[CMD] Warning: Extracted {len(frames)} frames, but target is {num_frames}")
            # If we have too few frames, duplicate the last frame
            while len(frames) < num_frames:
                frames.append(frames[-1].copy())
                frame_path = os.path.join(frames_dir, f"frame_{len(frames)-1:06d}.png")
                cv2.imwrite(frame_path, frames[-1])
            # If we have too many frames, truncate
            if len(frames) > num_frames:
                frames = frames[:num_frames]
        
        print(f"[CMD] Successfully extracted and processed {len(frames)} frames")
        
        # Verify frame files exist in temporary directory
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
        if len(frame_files) != num_frames:
            print(f"[CMD] Warning: Found {len(frame_files)} frame files but expected {num_frames}")
            # Ensure all frames exist by checking for gaps
            for i in range(num_frames):
                expected_frame = f"frame_{i:06d}.png"
                if expected_frame not in frame_files:
                    print(f"[CMD] Missing frame file: {expected_frame}, duplicating adjacent frame")
                    # Find the closest existing frame
                    closest_idx = min(max(0, i-1), len(frames)-1)
                    # Save a copy of the closest frame
                    cv2.imwrite(os.path.join(frames_dir, expected_frame), frames[closest_idx])
            # Recheck frame files
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            print(f"[CMD] After fix: {len(frame_files)} frame files")
        
        # Check if input video has audio - use direct FFprobe method
        print(f"[CMD] Checking if input video has audio: {input_video_path}")
        has_audio = check_video_has_audio(input_video_path)
        print(f"[CMD] Input video has audio: {has_audio}")
        
        temp_audio_file = None
        
        # Extract audio if present using ffmpeg directly
        if has_audio:
            temp_audio_file = os.path.join(output_folder, f"temp_audio_{timestamp}.aac")
            
            # Use ffmpeg to extract audio
            extract_cmd = [
                'ffmpeg', '-y',
                '-i', f'"{input_video_path}"',
                '-t', str(extract_duration_sec),
                '-vn',
                '-c:a', 'aac',
                '-b:a', '192k',
                f'"{temp_audio_file}"'
            ]
            
            print(f"[CMD] Extracting audio with command: {' '.join(extract_cmd)}")
            result = subprocess.run(' '.join(extract_cmd), shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            
            # Verify the audio file was created successfully
            if os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                print(f"[CMD] Successfully extracted audio to {temp_audio_file} (size: {os.path.getsize(temp_audio_file)} bytes)")
            else:
                print(f"[CMD] Failed to extract audio or audio file is empty")
                
                # Try alternative audio extraction methods
                for method_name, cmd_args in [
                    ("direct copy", ['-c:a', 'copy']),
                    ("mp3 format", ['-c:a', 'libmp3lame', '-q:a', '4']),
                    ("amr format", ['-c:a', 'libopencore_amrnb', '-ar', '8000', '-ab', '12.2k'])
                ]:
                    print(f"[CMD] Trying {method_name} audio extraction method")
                    if "mp3" in method_name:
                        alt_audio_file = os.path.splitext(temp_audio_file)[0] + ".mp3"
                    elif "amr" in method_name:
                        alt_audio_file = os.path.splitext(temp_audio_file)[0] + ".amr"
                    else:
                        alt_audio_file = temp_audio_file
                        
                    alt_cmd = [
                        'ffmpeg', '-y',
                        '-i', f'"{input_video_path}"',
                        '-t', str(extract_duration_sec),
                        '-vn'
                    ] + cmd_args + [f'"{alt_audio_file}"']
                    
                    subprocess.run(' '.join(alt_cmd), shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    
                    if os.path.exists(alt_audio_file) and os.path.getsize(alt_audio_file) > 0:
                        print(f"[CMD] Successfully extracted audio using {method_name}: {alt_audio_file}")
                        temp_audio_file = alt_audio_file
                        break
                    else:
                        print(f"[CMD] {method_name} extraction failed")
                
                if not os.path.exists(temp_audio_file) or os.path.getsize(temp_audio_file) == 0:
                    print(f"[CMD] All audio extraction methods failed")
                    temp_audio_file = None
        
        # Step 2: Encode the frames to a video
        # If we have audio, use a different approach that's more reliable
        if temp_audio_file and os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
            print(f"[CMD] Using two-pass encoding to preserve audio quality")
            
            # First create video without audio - using vsync 0 to preserve all frames
            temp_video_file = os.path.join(output_folder, f"temp_video_{timestamp}.mp4")
            video_cmd = [
                'ffmpeg', '-y',
                '-framerate', '16',
                '-i', f'"{os.path.join(frames_dir, "frame_%06d.png")}"',
                '-c:v', 'libx264',
                '-fps_mode', 'passthrough',
                '-profile:v', 'high',
                '-level', '3.1',
                '-preset', 'veryslow',
                '-crf', '12',
                '-pix_fmt', 'yuv420p',
                '-an',
                f'"{temp_video_file}"'
            ]
            
            print(f"[CMD] Creating video without audio...")
            subprocess.run(' '.join(video_cmd), shell=True)
            
            # Verify the video has the correct number of frames
            verify_cap = cv2.VideoCapture(temp_video_file)
            if verify_cap.isOpened():
                temp_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                verify_cap.release()
                
                if temp_frames != num_frames:
                    print(f"[CMD] Warning: Intermediate video has {temp_frames} frames, but expected {num_frames}")
                    
                    # Try again with an alternative method if frame count is incorrect
                    alt_video_cmd = [
                        'ffmpeg', '-y',
                        '-r', '16',
                        '-start_number', '0',
                        '-i', f'"{os.path.join(frames_dir, "frame_%06d.png")}"',
                        '-vframes', str(num_frames),
                        '-c:v', 'libx264',
                        '-fps_mode', 'passthrough',
                        '-pix_fmt', 'yuv420p',
                        '-an',
                        f'"{temp_video_file}"'
                    ]
                    
                    print(f"[CMD] Retrying with alternative encoding method...")
                    subprocess.run(' '.join(alt_video_cmd), shell=True)
            
            # Then combine video and audio
            final_cmd = [
                'ffmpeg', '-y',
                '-i', f'"{temp_video_file}"',
                '-i', f'"{temp_audio_file}"',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                f'"{reencoded_video}"'
            ]
            
            print(f"[CMD] Combining video and audio...")
            subprocess.run(' '.join(final_cmd), shell=True)
            
            # Clean up temporary video
            if os.path.exists(temp_video_file):
                try:
                    os.remove(temp_video_file)
                except Exception as e:
                    print(f"[CMD] Warning: Could not remove temp video file: {e}")
                
        else:
            # Standard encoding without audio - using vsync 0 to preserve all frames
            print(f"[CMD] Encoding video without audio (no valid audio detected)")
            video_cmd = [
                'ffmpeg', '-y',
                '-framerate', '16',
                '-i', f'"{os.path.join(frames_dir, "frame_%06d.png")}"',
                '-c:v', 'libx264',
                '-fps_mode', 'passthrough',
                '-vframes', str(num_frames),
                '-profile:v', 'high',
                '-level', '3.1',
                '-preset', 'veryslow',
                '-crf', '12',
                '-pix_fmt', 'yuv420p',
                '-an',
                f'"{reencoded_video}"'
            ]
            
            subprocess.run(' '.join(video_cmd), shell=True)
        
        # Verify the frame count and retry with different method if incorrect
        max_attempts = 3
        for attempt in range(max_attempts):
            verify_cap = cv2.VideoCapture(reencoded_video)
            if verify_cap.isOpened():
                output_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                output_fps = verify_cap.get(cv2.CAP_PROP_FPS)
                verify_cap.release()
                verify_cap = None
                
                print(f"[CMD] Re-encoded video has {output_frames} frames at {output_fps} FPS (target: {num_frames} frames at 16 FPS)")
                
                if output_frames == num_frames:
                    print(f"[CMD] Frame count verified successfully")
                    break
                    
                if attempt < max_attempts - 1:
                    print(f"[CMD] Incorrect frame count ({output_frames} vs {num_frames}). Retrying with alternative method...")
                    
                    # Try more aggressive approach to force exact frame count
                    retry_cmd = [
                        'ffmpeg', '-y',
                        '-r', '16',
                        '-start_number', '0',
                        '-i', f'"{os.path.join(frames_dir, "frame_%06d.png")}"',
                        '-vframes', str(num_frames),
                        '-c:v', 'libx264',
                        '-fps_mode', 'passthrough',
                        '-pix_fmt', 'yuv420p',
                        f'"{reencoded_video}"'
                    ]
                    
                    if temp_audio_file and os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                        # Include audio if available
                        retry_cmd = [
                            'ffmpeg', '-y',
                            '-r', '16',
                            '-start_number', '0',
                            '-i', f'"{os.path.join(frames_dir, "frame_%06d.png")}"',
                            '-i', f'"{temp_audio_file}"',
                            '-vframes', str(num_frames),
                            '-c:v', 'libx264',
                            '-c:a', 'aac',
                            '-b:a', '192k',
                            '-fps_mode', 'passthrough',
                            '-pix_fmt', 'yuv420p',
                            f'"{reencoded_video}"'
                        ]
                    
                    subprocess.run(' '.join(retry_cmd), shell=True)
                else:
                    print(f"[CMD] Warning: Could not achieve target frame count after {max_attempts} attempts")
            else:
                print(f"[CMD] Warning: Could not open re-encoded video for verification")
                break
        
        # If after all retries we still don't have the right frame count, create a new version from scratch
        final_verify_cap = cv2.VideoCapture(reencoded_video)
        if final_verify_cap.isOpened():
            final_frame_count = int(final_verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            final_verify_cap.release()
            
            if final_frame_count != num_frames:
                print(f"[CMD] Final verification failed: {final_frame_count} vs {num_frames}. Creating video directly from frames...")
                
                # Create a new video directly from the frames
                # This is a last resort measure that's more reliable but may have lower quality
                new_output = os.path.join(output_folder, f"reencoded_fixed_{timestamp}_{sanitized_name}{ext}")
                
                # Use OpenCV to create the video directly
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(new_output, fourcc, 16, (target_width, target_height))
                
                if out.isOpened():
                    for i in range(num_frames):
                        frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
                        if os.path.exists(frame_path):
                            frame = cv2.imread(frame_path)
                            if frame is not None:
                                out.write(frame)
                    
                    out.release()
                    print(f"[CMD] Created direct video with {num_frames} frames: {new_output}")
                    
                    # If we have audio, add it to the new video
                    if temp_audio_file and os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                        with_audio = add_audio_to_video(input_video_path, new_output, output_folder, temp_audio_file)
                        if with_audio[0] != new_output:
                            reencoded_video = with_audio[0]
                            print(f"[CMD] Using new video with audio: {reencoded_video}")
                        else:
                            reencoded_video = new_output
                    else:
                        reencoded_video = new_output
                else:
                    print(f"[CMD] Could not create direct video - keeping original re-encoded version")
        
        # Verify if output video has audio
        has_output_audio = check_video_has_audio(reencoded_video)
        print(f"[CMD] Output video has audio: {has_output_audio}")
        
        print(f"[CMD] Video re-encoded successfully: {reencoded_video}")
        return reencoded_video
    
    except Exception as e:
        if cap is not None:
            cap.release()
        if verify_cap is not None:
            verify_cap.release()
        print(f"[CMD] Error during video re-encoding: {e}")
        traceback.print_exc()
        return input_video_path
    
    finally:
        # Clean up temporary frames and audio
        try:
            # Clean up frame files
            if frames_dir and os.path.exists(frames_dir):
                frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame_")]
                for f in frame_files:
                    try:
                        os.remove(os.path.join(frames_dir, f))
                    except Exception as e:
                        print(f"[CMD] Warning: Could not remove frame file {f}: {e}")
                        
                # Try to remove the directory
                try:
                    os.rmdir(frames_dir)
                except Exception as e:
                    print(f"[CMD] Warning: Could not remove frames directory: {e}")
            
            # Clean up audio file
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                except Exception as e:
                    print(f"[CMD] Warning: Could not remove temp audio file: {e}")
                    
            # Clean up video file
            if temp_video_file and os.path.exists(temp_video_file):
                try:
                    os.remove(temp_video_file)
                except Exception as e:
                    print(f"[CMD] Warning: Could not remove temp video file: {e}")
                    
        except Exception as e:
            print(f"[CMD] Warning: Error during cleanup: {e}")
            traceback.print_exc()

def check_video_has_audio(video_path):
    """Check if a video file contains audio streams.

    ffprobe's stream list decides whenever ffprobe runs. The old third check looked for an FFmpeg
    error message that FFmpeg 9 no longer prints, so it reported audio in silent videos.
    Arguments are passed as a list, so paths with spaces or quotes work.
    """
    if not os.path.exists(video_path):
        print(f"[CMD] Video file not found: {video_path}")
        return False

    # Method 1: ffprobe stream list (authoritative)
    try:
        import json
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-select_streams', 'a', video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0 and result.stdout.strip():
            streams = json.loads(result.stdout).get('streams', [])
            if streams:
                print(f"[CMD] Audio streams detected: {len(streams)} in {video_path}")
            else:
                print(f"[CMD] No audio stream in {video_path}")
            return len(streams) > 0
        print(f"[CMD] ffprobe could not read {video_path}: {result.stderr.strip()[:300]}")
    except Exception as e:
        print(f"[CMD] Error in ffprobe audio detection: {e}")

    # Method 2: the stream list FFmpeg prints for its input
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-i', video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace'
        )
        if re.search(r"Stream #\d+:\d+.*: Audio:", result.stderr):
            print(f"[CMD] Audio stream found in {video_path} (FFmpeg stream list)")
            return True
    except Exception as e:
        print(f"[CMD] Error in FFmpeg audio detection: {e}")

    print(f"[CMD] No audio streams detected in {video_path}")
    return False

def add_audio_to_video(input_video_path, output_video_path, temp_dir=None, temp_audio_file=None):
    """
    Extract audio from input video and add it to the output video without re-encoding the video.
    
    Args:
        input_video_path: Path to the input video with audio
        output_video_path: Path to the output video that needs audio
        temp_dir: Optional temporary directory to store the audio file
        temp_audio_file: Optional pre-extracted audio file (to avoid re-extraction)
        
    Returns:
        Tuple of (path to the new video with audio, path to the temp audio file used)
        The video path could be the same as output_video_path if audio transfer failed
    """
    try:
        # If no temp_audio_file is provided, extract from input
        if temp_audio_file is None or not os.path.exists(temp_audio_file):
            # Check if input video has audio
            has_audio = check_video_has_audio(input_video_path)
            if not has_audio:
                print(f"[CMD] Input video has no audio to transfer: {input_video_path}")
                return output_video_path, None
            
            print(f"[CMD] Extracting audio from original input: {input_video_path}")
            
            # Create a timestamp for unique temp filenames
            timestamp = int(time.time())
            if temp_dir is None:
                temp_dir = "temp_videos"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create temporary audio file
            temp_audio_file = os.path.join(temp_dir, f"temp_audio_{timestamp}.aac")
            
            # Extract audio from input video - use higher verbosity to debug
            extract_cmd = [
                'ffmpeg', '-y', 
                '-i', f'"{input_video_path}"',  # Quoted path for Windows compatibility
                '-vn',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-v', 'info',
                f'"{temp_audio_file}"'  # Quoted path for Windows compatibility
            ]
            
            print(f"[CMD] Extracting audio with command: {' '.join(extract_cmd)}")
            result = subprocess.run(' '.join(extract_cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Log more details about extraction for debugging
            if result.stdout:
                print(f"[CMD] FFmpeg stdout: {result.stdout}")
            if result.stderr:
                print(f"[CMD] FFmpeg stderr: {result.stderr}")
            
            # Verify the audio file was created successfully
            if not os.path.exists(temp_audio_file) or os.path.getsize(temp_audio_file) == 0:
                print(f"[CMD] Failed to extract audio or audio file is empty")
                
                # Try multiple fallback methods in sequence
                fallback_methods = [
                    # Method 1: Copy stream without re-encoding
                    {
                        "name": "copy stream",
                        "cmd": [
                            'ffmpeg', '-y',
                            '-i', f'"{input_video_path}"',
                            '-vn',
                            '-c:a', 'copy',
                            '-v', 'info',
                            f'"{temp_audio_file}"'
                        ]
                    },
                    # Method 2: Try with mp3 codec instead
                    {
                        "name": "mp3 codec",
                        "cmd": [
                            'ffmpeg', '-y',
                            '-i', f'"{input_video_path}"',
                            '-vn',
                            '-c:a', 'libmp3lame',
                            '-q:a', '4',
                            '-v', 'info',
                            f'"{os.path.splitext(temp_audio_file)[0] + ".mp3"}"'
                        ]
                    },
                    # Method 3: Explicitly select the audio stream
                    {
                        "name": "explicit audio stream selection",
                        "cmd": [
                            'ffmpeg', '-y',
                            '-i', f'"{input_video_path}"',
                            '-map', '0:a:0',
                            '-c:a', 'aac',
                            '-b:a', '192k',
                            '-v', 'info',
                            f'"{temp_audio_file}"'
                        ]
                    },
                    # Method 4: Using amr codec for mobile videos
                    {
                        "name": "amr codec",
                        "cmd": [
                            'ffmpeg', '-y',
                            '-i', f'"{input_video_path}"',
                            '-vn',
                            '-ar', '8000',
                            '-ab', '12.2k',
                            '-c:a', 'libopencore_amrnb',
                            f'"{os.path.splitext(temp_audio_file)[0] + ".amr"}"'
                        ]
                    }
                ]
                
                for method in fallback_methods:
                    print(f"[CMD] Trying alternative audio extraction ({method['name']}): {' '.join(method['cmd'])}")
                    alt_result = subprocess.run(' '.join(method['cmd']), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    # For method 2 and 4, we have a different output file
                    if "mp3 codec" in method["name"]:
                        mp3_file = os.path.splitext(temp_audio_file)[0] + '.mp3'
                        if os.path.exists(mp3_file) and os.path.getsize(mp3_file) > 0:
                            print(f"[CMD] Alternative extraction succeeded with MP3: {mp3_file}")
                            temp_audio_file = mp3_file
                            break
                    elif "amr codec" in method["name"]:
                        amr_file = os.path.splitext(temp_audio_file)[0] + '.amr'
                        if os.path.exists(amr_file) and os.path.getsize(amr_file) > 0:
                            print(f"[CMD] Alternative extraction succeeded with AMR: {amr_file}")
                            temp_audio_file = amr_file
                            break
                    elif os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                        print(f"[CMD] Alternative extraction succeeded: {temp_audio_file} (size: {os.path.getsize(temp_audio_file)} bytes)")
                        break
                
                # Check if any method worked
                audio_success = False
                for ext in ['', '.mp3', '.amr']:
                    check_file = os.path.splitext(temp_audio_file)[0] + ext
                    if os.path.exists(check_file) and os.path.getsize(check_file) > 0:
                        temp_audio_file = check_file
                        audio_success = True
                        break
                
                if audio_success:
                    print(f"[CMD] Successfully extracted audio after fallback attempts: {temp_audio_file}")
                else:
                    print(f"[CMD] All audio extraction methods failed")
                    return output_video_path, None
            else:
                print(f"[CMD] Successfully extracted audio: {temp_audio_file} (size: {os.path.getsize(temp_audio_file)} bytes)")
        else:
            print(f"[CMD] Using pre-extracted audio file: {temp_audio_file} (size: {os.path.getsize(temp_audio_file)} bytes)")
            
            # Double-check that the temp audio file actually has content
            if os.path.getsize(temp_audio_file) == 0:
                print(f"[CMD] Pre-extracted audio file is empty, cannot use it")
                return output_video_path, None
        
        # Create final output with audio
        output_with_audio = os.path.splitext(output_video_path)[0] + "_with_audio.mp4"
        
        # Combine video and audio without re-encoding
        combine_cmd = [
            'ffmpeg', '-y',
            '-i', f'"{output_video_path}"',
            '-i', f'"{temp_audio_file}"',
            '-c:v', 'copy',
            '-c:a', 'aac', # Always convert to AAC for output compatibility
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            f'"{output_with_audio}"'
        ]
        
        print(f"[CMD] Adding audio without re-encoding: {' '.join(combine_cmd)}")
        result = subprocess.run(' '.join(combine_cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Log more details about the combination process
        if result.stderr:
            print(f"[CMD] FFmpeg stderr during audio addition: {result.stderr}")
        
        # Verify the output file exists and has appropriate size
        if os.path.exists(output_with_audio) and os.path.getsize(output_with_audio) > 0:
            # Verify the audio was actually added
            has_audio = check_video_has_audio(output_with_audio)
            if has_audio:
                print(f"[CMD] Successfully added audio to video: {output_with_audio}")
                return output_with_audio, temp_audio_file
            else:
                print(f"[CMD] Failed to add audio - output has no audio track. Trying fallback method.")
                
                # Try alternative method - direct copy without mapping
                alt_combine_cmd = [
                    'ffmpeg', '-y',
                    '-i', f'"{output_video_path}"',
                    '-i', f'"{temp_audio_file}"',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    f'"{output_with_audio}"'
                ]
                
                print(f"[CMD] Trying simplified audio addition: {' '.join(alt_combine_cmd)}")
                alt_result = subprocess.run(' '.join(alt_combine_cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Check if the alternative method worked
                if os.path.exists(output_with_audio) and os.path.getsize(output_with_audio) > 0:
                    has_audio = check_video_has_audio(output_with_audio)
                    if has_audio:
                        print(f"[CMD] Successfully added audio using simplified method")
                        return output_with_audio, temp_audio_file
                
                # If all fails, try one more approach with absolute paths
                try:
                    abs_output_path = os.path.abspath(output_video_path)
                    abs_audio_path = os.path.abspath(temp_audio_file)
                    abs_output_with_audio = os.path.abspath(output_with_audio)
                    
                    final_cmd = [
                        'ffmpeg', '-y',
                        '-i', f'"{abs_output_path}"',
                        '-i', f'"{abs_audio_path}"',
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-shortest',
                        f'"{abs_output_with_audio}"'
                    ]
                    
                    print(f"[CMD] Final attempt with absolute paths: {' '.join(final_cmd)}")
                    final_result = subprocess.run(' '.join(final_cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if os.path.exists(output_with_audio) and os.path.getsize(output_with_audio) > 0:
                        has_audio = check_video_has_audio(output_with_audio)
                        if has_audio:
                            print(f"[CMD] Successfully added audio using absolute paths")
                            return output_with_audio, temp_audio_file
                except Exception as e:
                    print(f"[CMD] Error in final audio addition attempt: {e}")
                
                print(f"[CMD] All audio addition methods failed. Keeping original without audio.")
                return output_video_path, temp_audio_file
        else:
            print(f"[CMD] Failed to add audio to video - output file doesn't exist or is empty")
            return output_video_path, temp_audio_file
            
    except Exception as e:
        print(f"[CMD] Error adding audio to video: {e}")
        traceback.print_exc()
        if temp_audio_file and os.path.exists(temp_audio_file):
            return output_video_path, temp_audio_file
        return output_video_path, None

def clean_temp_videos():
    """Clean up temporary videos that were created during re-encoding"""
    # Clean both the temp_videos and auto_pre_processed_videos folders
    folders_to_clean = ["temp_videos", "auto_pre_processed_videos"]
    
    for folder in folders_to_clean:
        if os.path.exists(folder) and os.path.isdir(folder):
            try:
                for file in os.listdir(folder):
                    # Clean up temporary files (reencoded videos, audio files, and frame directories)
                    if file.startswith("reencoded_") or \
                       file.startswith("temp_audio_") or \
                       file.startswith("temp_video_") or \
                       file.startswith("temp_frames_"):
                        
                        file_path = os.path.join(folder, file)
                        
                        # For directories, try to clean files inside first
                        if os.path.isdir(file_path) and file.startswith("temp_frames_"):
                            try:
                                frame_files = [f for f in os.listdir(file_path) if f.startswith("frame_")]
                                for frame in frame_files:
                                    try:
                                        frame_path = os.path.join(file_path, frame)
                                        os.remove(frame_path)
                                    except Exception as e:
                                        print(f"[CMD] Could not remove frame file {frame}: {e}")
                                        
                                # After removing files, try to remove the directory
                                try:
                                    os.rmdir(file_path)
                                    print(f"[CMD] Removed temporary directory: {file_path}")
                                except Exception as e:
                                    print(f"[CMD] Could not remove directory {file_path}: {e}")
                            except Exception as e:
                                print(f"[CMD] Error processing directory {file_path}: {e}")
                                
                        # For files, try to remove directly
                        elif os.path.isfile(file_path):
                            try:
                                # On Windows, files may be locked by another process
                                # Try multiple times with a short delay
                                max_attempts = 3
                                for attempt in range(max_attempts):
                                    try:
                                        os.remove(file_path)
                                        print(f"[CMD] Removed temporary file: {file_path}")
                                        break
                                    except PermissionError:
                                        if attempt < max_attempts - 1:
                                            print(f"[CMD] File {file_path} is locked, retrying in 1 second...")
                                            time.sleep(1)
                                        else:
                                            print(f"[CMD] Could not remove file {file_path} after {max_attempts} attempts")
                                    except Exception as e:
                                        print(f"[CMD] Error removing file {file_path}: {e}")
                                        break
                            except Exception as e:
                                print(f"[CMD] General error handling file {file_path}: {e}")
            except Exception as e:
                print(f"[CMD] Error cleaning {folder} folder: {e}") 
                
    print(f"[CMD] Temporary file cleanup complete") 