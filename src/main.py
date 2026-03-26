import cv2
import time
import numpy as np
from utils.logger import WriteLogs
from vision.preprocess import preprocess_frame
from vision.detect_objects import detect_objects
from depth.depth_estimation import Midas, Depth_Anything
from audio.audio_feedback import speak_command, start_audio, stop_audio
from navigation.roi_navigation import compute_roi_direction

def force_bgr_uint8(img):
    
    # Convert image to BGR uint8
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.astype(np.uint8)


def main():
    writer = WriteLogs()
    depth_model = Midas()
    start_audio()
    
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = 640, 480

    start_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_prep = preprocess_frame(frame)

        # YOLO detection
        detections, frame_yolo = detect_objects(frame_prep)

        # Depth estimation
        depth_colored, depth_map = depth_model.estimate_depth(frame_prep)
        # Resize depth map to match frame
        depth_map_resized = cv2.resize(depth_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        # ROI based navigation
        roi_x, roi_width, command= compute_roi_direction(
            depth_map_resized, frame_width, frame_height, detections
        )

        # Speak audio command
        command_start = time.time()
        speak_command(command)
        speak_time = time.time() - command_start

        # Draw translucent green ROI on original frame
        overlay = frame_prep.copy()
        alpha = 0.3  # transparency
        cv2.rectangle(overlay, (roi_x, 0), (roi_x + roi_width, frame_height), (0, 255, 0), -1)
        frame_display = cv2.addWeighted(overlay, alpha, frame_prep, 1 - alpha, 0)
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)

        # Prepare YOLO and depth frames for visualization
        frame_yolo = force_bgr_uint8(frame_yolo)
        frame_yolo = cv2.cvtColor(frame_yolo, cv2.COLOR_BGR2RGB)
        depth_colored = force_bgr_uint8(depth_colored)
        frame_yolo = cv2.resize(frame_yolo, (frame_width, frame_height))
        depth_colored = cv2.resize(depth_colored, (frame_width, frame_height))

        # Combine YOLO and depth views side by side
        combined = cv2.hconcat([frame_yolo, depth_colored])

        # Fps counter
        if ((time.time() - start_time) >= 1):
            writer.write_log(f"FPS: {fps} | Commmad: {command} | Depth Model: {depth_model.model_type}")
            fps = frame_count
            frame_count = 0
            start_time = time.time()
        else:
            frame_count += 1

        # FPS text on roi frame (frame_display)
        cv2.putText(
            frame_display,
            f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Show ROI overlay on original frame + combined view
        cv2.imshow("VisionSense Prototype - ROI Guidance", frame_display)
        cv2.imshow("YOLO + Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_audio()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
