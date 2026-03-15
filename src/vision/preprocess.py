import cv2
import numpy as np

def preprocess_frame(frame, width=640, height=480):

    # Resizes, normalizes, and enhances incoming frame.

    # Resize
    frame_resized = cv2.resize(frame, (width, height))
    
    # Convert to RGB (models expect RGB)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Brightness and contrast enhancement (for low light)
    frame_eq = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=20)

    return frame_eq