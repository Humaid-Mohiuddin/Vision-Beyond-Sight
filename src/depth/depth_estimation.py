import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class Midas:

    def __init__(self):
        # Load model (MiDaS model types)
        # self.model_type = "DPT_Large"     # MiDaS v3 - Large   (highest accuracy, slowest inference speed)
        self.model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # self.model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def estimate_depth(self, frame):
        
        # Returns depth map of the given frame.
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to("cuda" if torch.cuda.is_available() else "cpu")
    
        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()
            
        # Zero out pixels lesser than threshold value
        # Makes roi calculation less sensitive to distant objects
        # threshold = 1200
        # depth_map = np.array(depth_map)
        # depth_map[depth_map <= threshold] = 0
    
        # Normalize depth for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)

    
        return depth_colored, depth_map



class Depth_Anything:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_type = "Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(f"depth-anything/{self.model_type}")
        self.model = AutoModelForDepthEstimation.from_pretrained(f"depth-anything/{self.model_type}")
        self.model.to(self.device)
        self.model.eval()

    def estimate_depth(self, frame):
        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = depth.cpu().numpy()
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_PLASMA)

        return depth_colored, depth_map