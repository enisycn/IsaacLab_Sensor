import numpy as np
import av
import cv2
import os
import math
import base64

SIM_CROP_DIM = (200,300)
VIDEO_RESOLUTION = 16
GRID_SIZE = int(math.sqrt(VIDEO_RESOLUTION))
EPISODE_FRAME_LEN = 1002

def encode_image(image_path):
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
    

def save_grid_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

