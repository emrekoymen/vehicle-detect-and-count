"""
Realtime application for vehicle detection, tracking, and counting.
Combines YOLOv8 detection, ByteTrack tracking, and line-crossing counting.
Shows live video feed with annotations and saves counts to a CSV file.

CLI Example:
    python traffic-pipeline/realtime-app.py --weights traffic-pipeline/runs/yolov8/weights/best.pt --source path/to/video.mp4
    python traffic-pipeline/realtime-app.py --weights traffic-pipeline/runs/yolov8/weights/best.pt --source 0 (for webcam)
"""
import argparse
import cv2
import torch
import time
import csv
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO
from cjm_byte_track import BYTETracker # Assuming cjm_byte_track is installed
from shapely.geometry import LineString, Point

# Define class names based on dataset.yaml or your model training
# These should match the order from your dataset.yaml
CLASSES = ["car", "bus", "van", "others"] # Or load from model.names if available

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="traffic-pipeline/runs/yolov8/weights/best.pt",
                    help="Path to YOLOv8 model weights")
    ap.add_argument("--source", default="0", 
                    help="Video source (path to file, 0 for webcam, or RTSP stream)")
    ap.add_argument("--conf", type=float, default=0.3, 
                    help="Object detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, 
                    help="Intersection over Union (IoU) threshold for NMS")
    ap.add_argument("--output-csv", default=None,
                    help="Optional path to save counts CSV. Defaults to logs/counts_<timestamp>.csv")
    args = ap.parse_args()

    print(f"Using device: mps (Apple Silicon GPU)")
    device = 'mps'

    try:
        model = YOLO(args.weights)
        model.to(device) # Ensure model is on MPS
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print(f"Ensure weights path is correct: {args.weights}")
        return

    cap_source = 0 if args.source == "0" else args.source
    try:
        cap_source = int(cap_source) # Check if it's an integer for webcam index
    except ValueError:
        pass # Keep as string for file path or RTSP

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = BYTETracker(track_thresh=args.conf, match_thresh=0.8, track_buffer=30, frame_rate=30)

    # Define the counting line (adjust coordinates as needed for your video)
    # Example: a vertical line at x=frame_width/2, spanning the height
    line_x = frame_width // 3
    counting_line = LineString([(line_x, 0), (line_x, frame_height)])
    # For a horizontal line: LineString([(0, frame_height // 2), (frame_width, frame_height // 2)])
    print(f"Counting line defined from ({line_x},0) to ({line_x},{frame_height})")

    counted_ids = set()  # To store track IDs that have already been counted
    vehicle_counts = defaultdict(int) # Per-class counts

    # Output CSV setup
    log_dir = Path("traffic-pipeline/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.output_csv:
        csv_path = Path(args.output_csv)
    else:
        csv_path = log_dir / f"counts_{int(time.time())}.csv"
    
    print(f"Logging counts to: {csv_path}")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp_ms', 'class_id', 'class_name', 'track_id', 'total_for_class'])

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            # Perform detection
            # Forcing verbose=False as it can be quite spammy
            preds = model.predict(frame, conf=args.conf, iou=args.iou, device=device, verbose=False)
            
            # Check if predictor results are available and structured as expected
            if not preds or not hasattr(preds[0], 'boxes') or preds[0].boxes is None:
                cv2.imshow("Realtime Traffic Analysis", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                continue

            dets_xyxy = preds[0].boxes.xyxy.cpu() # Bounding boxes (x1, y1, x2, y2)
            scores = preds[0].boxes.conf.cpu()   # Confidence scores
            cls_ids = preds[0].boxes.cls.cpu()    # Class IDs

            # Update tracker
            if dets_xyxy.numel() > 0: # Check if there are any detections
                # Bytetracker expects: (x1, y1, x2, y2, score, class_id)
                # Ensure all tensors are on CPU and are numpy arrays for the tracker
                # Concatenate them correctly
                track_inputs = torch.cat([
                    dets_xyxy, 
                    scores.unsqueeze(1), 
                    cls_ids.unsqueeze(1)
                ], dim=1).numpy()
                online_targets = tracker.update(track_inputs, frame.shape) # Pass frame.shape for image size context
            else:
                online_targets = tracker.update_without_detection(frame.shape) # Keep tracker alive

            # Draw counting line
            cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2) # Red line

            current_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            for t in online_targets:
                x1, y1, x2, y2 = map(int, t[:4])
                track_id = int(t[4])
                class_id = int(t[5]) 
                # score = t[6] # if available from tracker output

                # Draw bounding box and track ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                label = f"ID:{track_id}"
                if class_id < len(CLASSES):
                    label += f" {CLASSES[class_id]}"
                else:
                    label += f" Cls:{class_id}" # Fallback if class_id is out of bounds
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Counting logic
                # Use center of the bottom edge of the bounding box for crossing detection
                # This is often more robust for vehicles than the centroid.
                # cx = (x1 + x2) / 2
                # cy = y2 # bottom-center y
                # Or use centroid:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                current_point = Point(cx, cy)

                if track_id not in counted_ids:
                    # Check if the point crosses the line. 
                    # This simple check assumes movement primarily perpendicular to the line.
                    # For more robust directionality, you'd need to store previous positions.
                    if current_point.crosses(counting_line):
                        vehicle_counts[class_id] += 1
                        counted_ids.add(track_id)
                        class_name = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
                        print(f"Vehicle Counted! ID: {track_id}, Class: {class_name} ({class_id}), Total {class_name}s: {vehicle_counts[class_id]}")
                        csv_writer.writerow([current_time_ms, class_id, class_name, track_id, vehicle_counts[class_id]])
                        csv_file.flush() # Write to disk immediately
            
            # Display counts on frame
            y_offset = 30
            for i, class_name in enumerate(CLASSES):
                count = vehicle_counts[i] # Access count by original class_id index
                cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                y_offset += 30

            cv2.imshow("Realtime Traffic Analysis", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"Total counts saved to {csv_path}")
        for i, class_name in enumerate(CLASSES):
             print(f"  {class_name}: {vehicle_counts[i]}")

if __name__ == "__main__":
    main()