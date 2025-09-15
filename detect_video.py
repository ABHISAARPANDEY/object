#!/usr/bin/env python3
"""
YOLO-CROWD Video Detection Script
Processes video and outputs a new video with crowd detection results
"""
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

def detect_video(input_video, weights, output_video="output_detection.mp4", conf_thres=0.25, iou_thres=0.45):
    """
    Process video with YOLO-CROWD detection
    
    Args:
        input_video: Path to input video file
        weights: Path to model weights
        output_video: Path to output video file
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
    """
    
    print("Loading YOLO-CROWD model...")
    device = torch.device('cpu')
    model = attempt_load(weights, map_location=device)
    model.eval()
    
    # Get model info
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    img_size = 640
    
    print("Processing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Show progress
        if frame_count % 50 == 0 or frame_count == 1:
            print(f"Processing frame {frame_count}/{total_frames}")
        
        # Prepare image for model
        img = cv2.resize(frame, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Process detections
        det = pred[0] if len(pred) > 0 else None
        if det is not None and len(det):
            # Rescale boxes from img_size to frame size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            
            # Draw bounding boxes
            for *xyxy, conf, cls in det:
                plot_one_box(xyxy, frame, label=None, color=colors[int(cls)], line_thickness=2)
        
        # Add crowd count text
        total_detections = len(det) if det is not None else 0
        cv2.putText(frame, f'People Count: {total_detections}', (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Show frame (optional - comment out if you don't want to see the video)
        cv2.imshow('YOLO-CROWD Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Processing complete!")
    print(f"📁 Output video saved as: {output_video}")
    print(f"📊 Total frames processed: {frame_count}")

if __name__ == "__main__":
    # Configuration
    input_video = "demo_video2.mp4"
    weights = "yolo-crowd.pt"
    output_video = "demo_video2_detected.mp4"
    
    print("🎯 YOLO-CROWD Video Detection")
    print("=" * 40)
    
    # Check if files exist
    import os
    if not os.path.exists(input_video):
        print(f"❌ Error: Input video '{input_video}' not found")
    elif not os.path.exists(weights):
        print(f"❌ Error: Model weights '{weights}' not found")
    else:
        detect_video(input_video, weights, output_video)
