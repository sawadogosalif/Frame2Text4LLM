from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np 

@dataclass
class VideoFrame:
    """
    A dataclass representing a video frame with associated metadata.
    """
    image: np.ndarray
    time: float
    frame_number: int
    
    @property
    def time_formatted(self) -> str:
        """
        Return the time in MM:SS:mmm format (minutes:seconds:milliseconds)
        """
        minutes = int(self.time // 60)
        seconds = int(self.time % 60)
        milliseconds = int((self.time % 1) * 1000)
        
        return f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    
    def to_dict(self) -> Dict:
        """Convert the frame to a dictionary format."""
        return {
            'image': self.image,
            'time': self.time,
            'time_formatted': self.time_formatted,
            'frame_number': self.frame_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoFrame':
        """Create a VideoFrame from a dictionary."""
        return cls(data['image'], data['time'], data['frame_number'])


@dataclass
class Video:
    """
    A dataclass representing video data and metadata.
    """
    path: str
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0
    frames: List[VideoFrame] = field(default_factory=list)
    subtitle_region: Optional[Tuple[int, int, int, int]] = None