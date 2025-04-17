
import cv2
import base64
import numpy as np


def encode_numpy_image(np_image, image_format=".jpg"):
    """Convert a NumPy array image to base64-encoded string."""
    success, buffer = cv2.imencode(image_format, np_image)
    if not success:
        raise ValueError("Image encoding failed")
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str