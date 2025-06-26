from .base import VideoFrame
from beartype.typing import List, Dict, Union, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from loguru import logger


def display_frames(frames: List[Union[VideoFrame, Dict]], max_samples: int = 4, 
                   show_subtitle_region: bool = True, 
                   subtitle_region: Tuple[int, int, int, int] = None) -> None:
    """
    Display a sample of extracted frames, with optional subtitle region highlighting.
    
    Args:
        frames: List of VideoFrame objects or dictionaries
        max_samples: Number of frames to display
        show_subtitle_region: Whether to highlight the subtitle region
        subtitle_region: Tuple (y1, y2, x1, x2) representing the subtitle region
    """
    if not frames:
        logger.info("No frames to display")
        return
    
    std_frames = []
    for frame in frames:
        if isinstance(frame, VideoFrame):
            std_frames.append(frame.to_dict())
        elif isinstance(frame, dict) and 'image' in frame:
            std_frames.append(frame)
        else:
            continue
    
    num_images = min(len(std_frames), max_samples)
    max_columns = 4
    num_columns = min(num_images, max_columns)
    num_rows = math.ceil(num_images / max_columns)
    logger.info(num_rows)
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(4 * num_columns, 4 * num_rows))
    axes = np.atleast_2d(axes)
    
    for idx, frame_info in enumerate(std_frames[:max_samples]):
        row = idx // max_columns
        col = idx % max_columns
        
        frame_rgb = cv2.cvtColor(frame_info['image'], cv2.COLOR_BGR2RGB)
        
        # Draw subtitle region if requested
        if show_subtitle_region and subtitle_region:
            y1, y2, x1, x2 = subtitle_region
            frame_copy = frame_rgb.copy()
            # OpenCV uses BGR, but we're working with RGB here
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            axes[row, col].imshow(frame_copy)
        else:
            axes[row, col].imshow(frame_rgb)
        
        if 'time_formatted' in frame_info:
            time_text = frame_info['time_formatted']
        else:
            time_text = f"{frame_info['time']:.2f}s"
            
        title = f"Time: {time_text}\nFrame #{frame_info['frame_number']}"
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # Hide unused axes
    for idx in range(num_images, num_rows * num_columns):
        row = idx // max_columns
        col = idx % max_columns
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()