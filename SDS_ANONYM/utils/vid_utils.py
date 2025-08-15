import numpy as np
import av
import cv2
import os
import math
import base64

SIM_CROP_DIM = (120,180)  # Reduced from (200,300) to lower resolution for GPT processing
VIDEO_RESOLUTION = 9   # 9 frames (3x3 grid) - good balance between quality and efficiency
GRID_SIZE = int(math.sqrt(VIDEO_RESOLUTION))
EPISODE_FRAME_LEN = 1002

def encode_image(image_path, max_size=None, quality=60):  # Reduced from quality=90 to 60
  """
  Encode an image file to base64 PNG.
  - max_size: optional (width, height) tuple to constrain max dimensions while preserving aspect ratio
  - quality: 0-100, mapped to PNG compression (0 = no compression, 9 = max). Lower quality = more compression for GPT
  """
  try:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
      # Fallback to raw file read if OpenCV cannot read
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    # Resize if max_size is provided
    if max_size is not None and isinstance(max_size, (tuple, list)) and len(max_size) == 2:
      max_w, max_h = int(max_size[0]), int(max_size[1])
      h, w = img.shape[:2]
      if w > 0 and h > 0:
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale < 1.0:
          new_w = max(1, int(w * scale))
          new_h = max(1, int(h * scale))
          img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Map quality (0-100) to PNG compression level (0-9)
    try:
      q = int(quality)
    except Exception:
      q = 90
    q = max(0, min(100, q))
    png_compression = max(0, min(9, int(round((100 - q) / 10.0))))

    success, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    if not success:
      # Fallback to raw file read
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    return base64.b64encode(buf.tobytes()).decode('utf-8')

  except Exception:
    # Last-resort fallback
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def readVideoPyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    

def read_video(filename,resolution,training_fixed_length=False):

    container = av.open(filename)
    total_frames = container.streams.video[0].frames
    
    if training_fixed_length:
        indices = np.arange(1,EPISODE_FRAME_LEN,EPISODE_FRAME_LEN/VIDEO_RESOLUTION).astype(int)
    elif resolution == 0:
        indices = np.arange(1,total_frames,1).astype(int)
    else:
        indices = np.arange(1, total_frames, total_frames / resolution).astype(int)
    frames = readVideoPyav(container, indices)

    return frames

def crop_to_dim(input,toHeight=SIM_CROP_DIM[0],toWidth=SIM_CROP_DIM[1]):
    is_img = len(input.shape) == 3
    if is_img:
        # Image
        input = input.reshape(1,*input.shape)
    
    _,height,width,_ = input.shape

    center = max(toHeight,height)/2, max(toWidth,width)/2
    x = int(center[1] - toWidth/2)
    y = int(center[0] - toHeight/2)
    frames = np.array([frame[y:y+toHeight, x:x+toWidth] for frame in input])
    
    if is_img:
        frames = frames.squeeze()
        
    return frames

def crop_upper_half(input):
    """
    Crop to keep only the upper half of frames for better gait analysis.
    Focuses on torso and upper body movements, removing legs/feet.
    """
    is_img = len(input.shape) == 3
    if is_img:
        input = input.reshape(1,*input.shape)
    
    _,height,width,_ = input.shape
    
    # Keep only upper 50% of the image (remove bottom half)
    crop_height = height // 2
    
    # Crop the upper half from each frame
    cropped_frames = []
    for frame in input:
        # Keep upper half only (torso and upper body focus)
        upper_half = frame[:crop_height, :, :]
        cropped_frames.append(upper_half)
    
    frames = np.array(cropped_frames)
    
    if is_img:
        frames = frames.squeeze()
        
    return frames

def crop_robot_focus(input, zoom_factor=1.5):
    """
    Crop to focus on the robot by removing sky area and magnifying.
    Crops from the upper portion and zooms in on the robot area.
    """
    is_img = len(input.shape) == 3
    if is_img:
        input = input.reshape(1,*input.shape)
    
    _,height,width,_ = input.shape
    
    # Remove top 40% (sky area) and crop to robot-focused area
    crop_top = int(height * 0.4)  # Remove top 40% (sky)
    crop_height = int(height * 0.5)  # Keep 50% of height (robot area)
    crop_width = width  # Keep full width
    
    # Crop the robot area first
    cropped_frames = []
    for frame in input:
        # Crop to robot area (remove sky)
        robot_area = frame[crop_top:crop_top+crop_height, :, :]
        
        # Resize to zoom in on robot (magnify)
        target_height = int(crop_height * zoom_factor)
        target_width = int(crop_width * zoom_factor)
        
        # Use OpenCV to resize (zoom)
        zoomed = cv2.resize(robot_area, (target_width, target_height))
        
        # If zoomed image is larger than original, crop to center
        if target_height > height or target_width > width:
            start_y = max(0, (target_height - height) // 2)
            start_x = max(0, (target_width - width) // 2)
            final_frame = zoomed[start_y:start_y+height, start_x:start_x+width]
        else:
            # If smaller, pad to original size
            pad_y = (height - target_height) // 2
            pad_x = (width - target_width) // 2
            final_frame = np.zeros((height, width, 3), dtype=np.uint8)
            final_frame[pad_y:pad_y+target_height, pad_x:pad_x+target_width] = zoomed
        
        cropped_frames.append(final_frame)
    
    frames = np.array(cropped_frames)
    
    if is_img:
        frames = frames.squeeze()
        
    return frames


def create_grid_image(video_path, grid_size=(GRID_SIZE, GRID_SIZE), margin=10, crop = True, crop_option='Full',training_fixed_length = False):
    frames = read_video(video_path,(grid_size[0] * grid_size[1]),training_fixed_length)
    h, w, _ = frames[0].shape
    
    if crop:
        if crop_option == "Full":
            frames = crop_to_dim(frames)
        elif crop_option == "Med":
            frames = crop_to_dim(frames,toHeight=int(2*SIM_CROP_DIM[0]),toWidth=int(2.2*SIM_CROP_DIM[1]))
        elif crop_option == "Vertical":
            frames = crop_to_dim(frames,toWidth=w)
        elif crop_option == "Horizontal":
            frames = crop_to_dim(frames,toHeight=h)
        elif crop_option == "RobotFocus":
            frames = crop_robot_focus(frames, zoom_factor=1.5)
        elif crop_option == "UpperHalf":
            frames = crop_upper_half(frames)

    h, w, _ = frames[0].shape
    grid_h = h * grid_size[0] + margin * (grid_size[0] - 1)
    grid_w = w * grid_size[1] + margin * (grid_size[1] - 1)
    grid_image = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # Create a white canvas

    for i, frame in enumerate(frames):
        row = i // grid_size[1]
        col = i % grid_size[1]
        y = row * (h + margin)
        x = col * (w + margin)
        grid_image[y:y+h, x:x+w, :] = frame
        
    black = np.zeros_like(frame)
    
    for j in range(1,grid_size[0]*grid_size[1] - i):
        row = (i+j) // grid_size[1]
        col = (i+j) % grid_size[1]
        y = row * (h + margin)
        x = col * (w + margin)
        grid_image[y:y+h, x:x+w, :] = black
    
    return grid_image

def gen_placehold_image(grid_size=(GRID_SIZE, GRID_SIZE), margin=10):
    h, w, _ = SIM_CROP_DIM + (3,)
    grid_h = h * grid_size[0] + margin * (grid_size[0] - 1)
    grid_w = w * grid_size[1] + margin * (grid_size[1] - 1)
    grid_image = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # Create a white canvas
        
    black = np.zeros(SIM_CROP_DIM + (3,))
    
    for i in range(0,grid_size[0]*grid_size[1]):
        row = (i) // grid_size[1]
        col = (i) % grid_size[1]
        y = row * (h + margin)
        x = col * (w + margin)
        grid_image[y:y+h, x:x+w, :] = black
    
    return grid_image 
    

def save_grid_image(image, output_path, quality=60):  # Reduced from quality=85 to 60
    """Save grid image with lower compression quality for reduced GPT processing burden"""
    if output_path.endswith('.png'):
        # For PNG, use higher compression level (0-9, where 9 is max compression)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 8])  # Increased from 6 to 8
    elif output_path.endswith('.jpg') or output_path.endswith('.jpeg'):
        # For JPEG, use lower quality (0-100, where 100 is best quality)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        # Default behavior for other formats
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

