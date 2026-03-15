import torch
import cv2
import numpy as np

# Load model (MiDaS small)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cuda" if torch.cuda.is_available() else "cpu")
midas.eval()

# Load transforms
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def estimate_depth(frame):
    
    # Returns depth map of the given frame.
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        prediction = midas(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()

    # Normalize depth for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)

    return depth_colored, depth_map
