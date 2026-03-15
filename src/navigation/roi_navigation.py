import numpy as np

def compute_roi_direction(detections, depth_map, frame_width, frame_height, roi_width_ratio=0.2, step=10):
    
    """
    Compute the optimal vertical ROI position and direction command.
    
    detections: list of YOLO detections [x1, y1, x2, y2, conf, class_id]
    depth_map: normalized depth map (0=near, 1=far)
    frame_width, frame_height: frame size
    roi_width_ratio: width of ROI relative to frame width (default 20%)
    step: sliding window step in pixels
    
    Returns:
        roi_x: x-coordinate of optimal ROI (left side)
        command: 'Move left', 'Move right', 'Path clear'
    """
    
    roi_width = int(frame_width * roi_width_ratio)
    min_cost = float('inf')
    best_x = 0

    # Preprocess detections into bounding boxes
    boxes = []
    if detections is not None:
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            boxes.append((int(x1), int(y1), int(x2), int(y2)))

    # Slide ROI across frame
    for x in range(0, frame_width - roi_width, step):
        roi_rect = (x, 0, x + roi_width, frame_height)

        # Cost 1: number of objects overlapping ROI
        overlap_count = 0
        for bx1, by1, bx2, by2 in boxes:
            # Check if box intersects ROI
            if bx2 >= roi_rect[0] and bx1 <= roi_rect[2]:
                overlap_count += 1

        # Cost 2: average depth value in ROI (closer objects = higher cost)
        roi_depth = depth_map[:, roi_rect[0]:roi_rect[2]]
        avg_depth_cost = np.mean(roi_depth)  # closer = higher

        total_cost = avg_depth_cost # weight objects more
        if total_cost < min_cost:
            min_cost = total_cost
            best_x = x

    # Determine command based on ROI position relative to frame center
    roi_center = best_x + roi_width // 2
    frame_center = frame_width // 2
    threshold = frame_width * 0.05  # 5% tolerance
    if roi_center < frame_center - threshold:
        command = "Move left"
    elif roi_center > frame_center + threshold:
        command = "Move right"
    else:
        command = "Path clear"

    return best_x, roi_width, command
