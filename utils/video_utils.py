#utils/video_utils.py
import cv2
import base64
import os
import math

def extract_frames_as_base64(video_path: str, frames_per_second: int = 2, max_dim: int = 960) -> list[str]:
    """
    Extracts frames from a video at a specific rate and converts them to Base64 strings.
    
    Args:
        video_path (str): Path to the input video file.
        frames_per_second (int): Number of frames to extract per second of video.
                                 Defaults to 2.
    
    Returns:
        list: A list of Base64 encoded strings (representing JPEG images).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file. Check format or permissions.")

    # Get the original video's Frames Per Second (FPS)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the "hop" (interval): how many frames to skip to match desired FPS
    # e.g., if video is 30fps and we want 2fps, we read every 15th frame (30 / 2)
    hop = math.floor(video_fps / frames_per_second)
    hop = max(1, hop)

    base64_frames = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break  # End of video

            # Only process the frame if it matches our hop interval
            if frame_count % hop == 0:

                h, w = frame.shape[:2]
                
                # Check if the longest side exceeds max_dim
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                # 1. Encode frame to JPEG format (reduces size for Base64)
                #    img_encode is a tuple: (success, encoded_image_buffer)
                success, buffer = cv2.imencode('.jpg', frame)
                
                if success:
                    # 2. Convert to bytes
                    frame_bytes = buffer.tobytes()
                    
                    # 3. Encode bytes to Base64 string
                    b64_string = base64.b64encode(frame_bytes).decode('utf-8')
                    base64_frames.append(b64_string)

            frame_count += 1
            
    finally:
        cap.release()

    print(f"Extracted {len(base64_frames)} frames from {video_path} (Original FPS: {video_fps:.2f})")
    return base64_frames


def video_base64_encoding(video_path: str) -> str:
    with open(video_path, "rb") as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode("utf-8")
    return video_base64
            