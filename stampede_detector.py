#!/usr/bin/env python3
"""
YOLO-CROWD Stampede Detection System
Advanced crowd analysis with stampede detection capabilities
"""
import cv2
import torch
import numpy as np
from collections import deque
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import math

class StampedeDetector:
    def __init__(self, weights_path, conf_thres=0.25, iou_thres=0.45):
        """Initialize the stampede detection system"""
        
        # Load YOLO-CROWD model
        print("Loading YOLO-CROWD model...")
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()
        
        # Move model to GPU for faster inference
        if self.device.type == 'cuda':
            self.model = self.model.cuda()
            print("✅ Model loaded on GPU for real-time processing!")
        else:
            print("⚠️  GPU not available, using CPU (slower)")
            # Optimize for CPU processing
            torch.set_num_threads(4)  # Use multiple CPU cores
            print("🔧 Optimized for CPU processing with 4 threads")
        
        # Model parameters - optimized for speed
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = 416  # Smaller image size for faster processing
        
        # Get model info
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        # Enhanced stampede detection parameters for better accuracy
        self.density_threshold = 1.8  # High density threshold (people per 100 pixels) - more sensitive
        self.movement_threshold = 6.0  # Movement speed threshold (pixels per frame) - more sensitive
        self.direction_consistency_threshold = 0.6  # Direction consistency threshold - more sensitive
        self.danger_zone_size = 120  # Size of danger zone analysis (pixels) - smaller zones for precision
        self.local_density_radius = 80  # Radius for local density calculation
        self.movement_history_length = 15  # Track more frames for better movement analysis
        
        # Enhanced tracking variables for better accuracy
        self.previous_detections = []
        self.movement_history = deque(maxlen=self.movement_history_length)  # Track more frames
        self.density_history = deque(maxlen=50)  # Track density over longer time
        self.alert_history = deque(maxlen=10)  # Track more recent alerts
        self.velocity_history = deque(maxlen=20)  # Track velocity patterns
        self.direction_history = deque(maxlen=15)  # Track direction changes
        
        # Statistics
        self.frame_count = 0
        self.total_people_detected = 0
        self.stampede_events = 0
        self.high_density_events = 0
        
    def calculate_crowd_density(self, detections, frame_shape):
        """Calculate crowd density in different regions of the frame"""
        if detections is None or len(detections) == 0:
            return 0.0, np.array([])
        
        height, width = frame_shape[:2]
        density_map = np.zeros((height // self.danger_zone_size, width // self.danger_zone_size))
        
        # Create density grid
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Map to grid
            grid_x = int(center_x // self.danger_zone_size)
            grid_y = int(center_y // self.danger_zone_size)
            
            # Ensure within bounds
            grid_x = min(grid_x, density_map.shape[1] - 1)
            grid_y = min(grid_y, density_map.shape[0] - 1)
            
            density_map[grid_y, grid_x] += 1
        
        # Calculate overall density
        total_area = (height * width) / (self.danger_zone_size ** 2)
        overall_density = len(detections) / total_area
        
        return overall_density, density_map
    
    def analyze_movement_patterns(self, current_detections, previous_detections):
        """Analyze movement patterns between frames"""
        if (previous_detections is None or len(previous_detections) == 0 or 
            current_detections is None or len(current_detections) == 0):
            return [], 0.0, 0.0
        
        movements = []
        total_movement = 0.0
        direction_vectors = []
        
        # Simple tracking: match detections by proximity
        for curr_det in current_detections:
            curr_center = ((curr_det[0] + curr_det[2]) / 2, (curr_det[1] + curr_det[3]) / 2)
            min_distance = float('inf')
            best_match = None
            
            for prev_det in previous_detections:
                prev_center = ((prev_det[0] + prev_det[2]) / 2, (prev_det[1] + prev_det[3]) / 2)
                distance = math.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                
                if distance < min_distance and distance < 50:  # Max tracking distance
                    min_distance = distance
                    best_match = prev_det
            
            if best_match is not None:
                prev_center = ((best_match[0] + best_match[2]) / 2, (best_match[1] + best_match[3]) / 2)
                movement = math.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                movements.append(movement)
                total_movement += movement
                
                # Calculate direction vector
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                if movement > 0:
                    direction_vectors.append((dx/movement, dy/movement))
        
        # Calculate average movement speed
        avg_movement = total_movement / len(movements) if movements else 0.0
        
        # Calculate direction consistency
        direction_consistency = 0.0
        if len(direction_vectors) > 1:
            # Calculate how consistent the movement directions are
            direction_sum = np.array([0.0, 0.0])
            for dx, dy in direction_vectors:
                direction_sum += np.array([dx, dy])
            
            avg_direction = direction_sum / len(direction_vectors)
            direction_magnitude = np.linalg.norm(avg_direction)
            direction_consistency = direction_magnitude
        
        return movements, avg_movement, direction_consistency
    
    def detect_stampede_conditions(self, density, avg_movement, direction_consistency, density_map):
        """Enhanced stampede detection with pattern analysis"""
        alerts = []
        danger_level = 0
        
        # Enhanced density analysis
        if density > self.density_threshold:
            alerts.append("HIGH DENSITY")
            danger_level += 2
            self.high_density_events += 1
        
        # Enhanced movement analysis with pattern recognition
        if avg_movement > self.movement_threshold:
            alerts.append("RAPID MOVEMENT")
            danger_level += 2
        
        # Check for acceleration patterns (sudden speed increase)
        if len(self.velocity_history) >= 5:
            recent_velocities = list(self.velocity_history)[-5:]
            if len(recent_velocities) >= 3:
                acceleration = recent_velocities[-1] - recent_velocities[0]
                if acceleration > 3.0:  # Sudden acceleration
                    alerts.append("ACCELERATION SPIKE")
                    danger_level += 2
        
        # Enhanced directional analysis
        if direction_consistency > self.direction_consistency_threshold and avg_movement > 2.0:
            alerts.append("DIRECTIONAL RUSH")
            danger_level += 3
        
        # Check for direction changes (panic behavior)
        if len(self.direction_history) >= 3:
            recent_directions = list(self.direction_history)[-3:]
            direction_variance = np.var(recent_directions)
            if direction_variance > 0.3:  # High direction variance indicates panic
                alerts.append("PANIC MOVEMENT")
                danger_level += 2
        
        # Enhanced local density hotspots with gradient analysis
        if density_map.size > 0:
            max_local_density = np.max(density_map)
            if max_local_density > 6:  # Lowered threshold for more sensitivity
                alerts.append("DENSITY HOTSPOT")
                danger_level += 1
            
            # Check for density gradients (crowd compression)
            if density_map.shape[0] > 1 and density_map.shape[1] > 1:
                density_gradient = np.gradient(density_map)
                max_gradient = np.max(np.abs(density_gradient))
                if max_gradient > 2.0:  # High density gradient
                    alerts.append("CROWD COMPRESSION")
                    danger_level += 1
        
        # Enhanced danger level determination with temporal analysis
        if len(self.alert_history) >= 3:
            recent_alerts = list(self.alert_history)[-3:]
            if len(recent_alerts) >= 2:
                # If multiple alerts in recent frames, increase danger
                if len([a for a in recent_alerts if len(a) > 0]) >= 2:
                    danger_level += 1
        
        # Store current alerts for temporal analysis
        self.alert_history.append(alerts)
        
        # Enhanced danger level thresholds
        if danger_level >= 7:
            danger_status = "CRITICAL STAMPEDE RISK"
            self.stampede_events += 1
        elif danger_level >= 5:
            danger_status = "HIGH STAMPEDE RISK"
        elif danger_level >= 3:
            danger_status = "MODERATE RISK"
        elif danger_level >= 1:
            danger_status = "LOW RISK"
        else:
            danger_status = "NORMAL"
        
        return alerts, danger_status, danger_level
    
    def get_local_density(self, center_x, center_y, detections, radius=None):
        """Enhanced local density calculation with adaptive radius"""
        if detections is None or len(detections) == 0:
            return 0
        
        if radius is None:
            radius = self.local_density_radius
        
        # Use numpy for faster calculations
        detections_array = np.array(detections)
        centers_x = (detections_array[:, 0] + detections_array[:, 2]) / 2
        centers_y = (detections_array[:, 1] + detections_array[:, 3]) / 2
        
        # Vectorized distance calculation
        distances = np.sqrt((centers_x - center_x)**2 + (centers_y - center_y)**2)
        count = np.sum(distances <= radius)
        
        # Weight by distance (closer people count more)
        weights = np.maximum(0, 1 - distances / radius)
        weighted_count = np.sum(weights[distances <= radius])
        
        return int(weighted_count)
    
    def draw_analysis_overlay(self, frame, detections, density_map, alerts, danger_status, danger_level, 
                            density, avg_movement, direction_consistency):
        """Draw analysis overlay on frame"""
        
        # Draw bounding boxes with enhanced individual density coloring
        if detections is not None and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2 = detection[:4]
                conf = detection[4]
                cls = int(detection[5])
                
                # Calculate local density for this specific detection
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                local_density = self.get_local_density(center_x, center_y, detections)
                
                # Enhanced color scheme based on individual local density
                if local_density > 6:  # High local density
                    color = (0, 0, 255)  # Red for high density
                elif local_density > 3:  # Medium local density
                    color = (0, 165, 255)  # Orange for medium density
                else:
                    color = (0, 255, 0)  # Green for normal density
                
                plot_one_box([x1, y1, x2, y2], frame, label=None, color=color, line_thickness=2)
        
        # Remove density heatmap overlay - keep normal crowd detection
        # (Commented out to avoid red overlay on crowd)
        
        # Draw danger status with enhanced levels
        if danger_status == "CRITICAL STAMPEDE RISK":
            status_color = (0, 0, 255)  # Red for critical stampede
            cv2.putText(frame, f"🚨 STAMPEDE ALERT: {danger_status}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        elif danger_status == "HIGH STAMPEDE RISK":
            status_color = (0, 165, 255)  # Orange for high risk
            cv2.putText(frame, f"⚠️ HIGH RISK: {danger_status}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        elif danger_status == "MODERATE RISK":
            status_color = (0, 255, 255)  # Yellow for moderate risk
            cv2.putText(frame, f"⚠️ MODERATE RISK: {danger_status}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        elif danger_status == "LOW RISK":
            status_color = (0, 255, 0)  # Green for low risk
            cv2.putText(frame, f"ℹ️ LOW RISK: {danger_status}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            # Don't show any danger text for normal situations
            pass
        
        # Draw metrics
        y_offset = 70
        cv2.putText(frame, f"People Count: {len(detections)}", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Density: {density:.3f}", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Movement: {avg_movement:.2f} px/frame", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Direction Consistency: {direction_consistency:.2f}", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw alerts
        if alerts:
            y_offset += 50
            cv2.putText(frame, "ALERTS:", (30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            for alert in alerts:
                y_offset += 30
                cv2.putText(frame, f"- {alert}", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw frame info
        cv2.putText(frame, f"Frame: {self.frame_count}", (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw color legend
        height, width = frame.shape[:2]
        legend_y = height - 120
        cv2.putText(frame, "Density Legend:", (30, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Green: Normal", (30, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Orange: Medium", (30, legend_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.putText(frame, "Red: High Density", (30, legend_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame
    
    def draw_persistent_boxes(self, frame, detections, density, alerts, danger_status, danger_level):
        """Draw persistent boxes with enhanced individual density coloring"""
        if detections is not None and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2 = detection[:4]
                
                # Calculate local density for this specific detection
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                local_density = self.get_local_density(center_x, center_y, detections)
                
                # Enhanced color scheme based on individual local density
                if local_density > 6:  # High local density
                    color = (0, 0, 255)  # Red for high density
                elif local_density > 3:  # Medium local density
                    color = (0, 165, 255)  # Orange for medium density
                else:
                    color = (0, 255, 0)  # Green for normal density
                
                plot_one_box([x1, y1, x2, y2], frame, label=None, color=color, line_thickness=2)
        
        # Draw danger status - only show red text for actual stampede situations
        if danger_status == "CRITICAL STAMPEDE RISK":
            status_color = (0, 0, 255)  # Red only for critical stampede
            cv2.putText(frame, f"🚨 STAMPEDE ALERT: {danger_status}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        elif danger_status == "HIGH STAMPEDE RISK":
            status_color = (0, 165, 255)  # Orange for high risk
            cv2.putText(frame, f"⚠️ HIGH RISK: {danger_status}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        # Draw metrics
        y_offset = 70
        cv2.putText(frame, f"People Count: {len(detections) if detections is not None else 0}", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Density: {density:.3f}", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Movement: 0.00 px/frame", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Direction Consistency: 0.00", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw alerts
        if alerts:
            y_offset += 50
            cv2.putText(frame, "ALERTS:", (30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            for alert in alerts:
                y_offset += 30
                cv2.putText(frame, f"- {alert}", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw frame info
        cv2.putText(frame, f"Frame: {self.frame_count}", (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw color legend
        height, width = frame.shape[:2]
        legend_y = height - 120
        cv2.putText(frame, "Density Legend:", (30, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Green: Normal", (30, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Orange: Medium", (30, legend_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.putText(frame, "Red: High Density", (30, legend_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def process_frame(self, frame):
        """Process a single frame for stampede detection"""
        self.frame_count += 1
        
        # Prepare image for model
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # Process detections
        detections = []
        if len(pred) > 0 and pred[0] is not None:
            det = pred[0]
            # Rescale boxes from img_size to frame size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            detections = det.cpu().numpy()
        
        # Calculate crowd density
        density, density_map = self.calculate_crowd_density(detections, frame.shape)
        self.density_history.append(density)
        
        # Enhanced movement analysis for better accuracy
        movements, avg_movement, direction_consistency = self.analyze_movement_patterns(
            detections, self.previous_detections)
        
        # Store movement data for pattern analysis
        self.movement_history.append(avg_movement)
        self.velocity_history.append(avg_movement)
        self.direction_history.append(direction_consistency)
        
        # Store current detections for next frame
        self.previous_detections = detections.copy() if detections is not None else []
        
        # Detect stampede conditions
        alerts, danger_status, danger_level = self.detect_stampede_conditions(
            density, avg_movement, direction_consistency, density_map)
        
        # Update statistics
        self.total_people_detected += len(detections) if detections is not None else 0
        
        # Draw analysis overlay
        processed_frame = self.draw_analysis_overlay(
            frame, detections, density_map, alerts, danger_status, danger_level,
            density, avg_movement, direction_consistency)
        
        return processed_frame, {
            'people_count': len(detections) if detections is not None else 0,
            'density': density,
            'avg_movement': avg_movement,
            'direction_consistency': direction_consistency,
            'alerts': alerts,
            'danger_status': danger_status,
            'danger_level': danger_level
        }
    
    def get_statistics(self):
        """Get detection statistics"""
        return {
            'total_frames': self.frame_count,
            'total_people_detected': self.total_people_detected,
            'stampede_events': self.stampede_events,
            'high_density_events': self.high_density_events,
            'avg_density': np.mean(self.density_history) if self.density_history else 0.0
        }

def detect_stampede_video(input_video, weights, output_video="stampede_analysis.mp4"):
    """Process video with stampede detection"""
    
    print("🎯 YOLO-CROWD Stampede Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = StampedeDetector(weights)
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"\n🔍 Starting Stampede Analysis...")
    
    # Process frames
    frame_count = 0
    analysis_results = []
    skip_frames = 2  # Process every 3rd frame for speed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Show progress
        if frame_count % 50 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            print(f"   Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # Process every nth frame for speed, but keep detections persistent
        if frame_count % (skip_frames + 1) == 0:
            processed_frame, analysis = detector.process_frame(frame)
            analysis_results.append(analysis)
            # Store the last detection results for persistence
            last_detections = detector.previous_detections
            last_density = analysis['density']
            last_alerts = analysis['alerts']
            last_danger_status = analysis['danger_status']
            last_danger_level = analysis['danger_level']
        else:
            # Use last detection results to maintain persistent boxes
            processed_frame = frame.copy()
            if 'last_detections' in locals() and last_detections is not None:
                # Draw persistent boxes from last detection
                detector.draw_persistent_boxes(processed_frame, last_detections, last_density, 
                                             last_alerts, last_danger_status, last_danger_level)
            
            analysis_results.append({
                'people_count': len(last_detections) if 'last_detections' in locals() and last_detections is not None else 0,
                'density': last_density if 'last_density' in locals() else 0.0,
                'avg_movement': 0.0,
                'direction_consistency': 0.0,
                'alerts': last_alerts if 'last_alerts' in locals() else [],
                'danger_status': last_danger_status if 'last_danger_status' in locals() else 'NORMAL',
                'danger_level': last_danger_level if 'last_danger_level' in locals() else 0
            })
        
        # Write frame to output video
        out.write(processed_frame)
        
        # Show frame (optional)
        cv2.imshow('Stampede Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("   ⏹️  Processing stopped by user")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Generate analysis report
    stats = detector.get_statistics()
    
    print(f"\n✅ Analysis Complete!")
    print(f"📁 Output video: {output_video}")
    print(f"📊 Analysis Results:")
    print(f"   Total frames processed: {stats['total_frames']}")
    print(f"   Total people detected: {stats['total_people_detected']}")
    print(f"   Stampede events: {stats['stampede_events']}")
    print(f"   High density events: {stats['high_density_events']}")
    print(f"   Average crowd density: {stats['avg_density']:.3f}")
    
    # Find critical moments
    critical_frames = []
    for i, result in enumerate(analysis_results):
        if result['danger_level'] >= 3:
            critical_frames.append((i, result))
    
    if critical_frames:
        print(f"\n⚠️  Critical Moments Detected:")
        for frame_num, result in critical_frames:
            timestamp = frame_num / fps
            print(f"   Frame {frame_num} ({timestamp:.1f}s): {result['danger_status']}")
            if result['alerts']:
                print(f"      Alerts: {', '.join(result['alerts'])}")
    
    return analysis_results, stats

if __name__ == "__main__":
    # Configuration
    input_video = "demo_video2.mp4"
    weights = "yolo-crowd.pt"
    output_video = "demo_video2_stampede_analysis.mp4"
    
    # Check if files exist
    import os
    if not os.path.exists(input_video):
        print(f"❌ Error: Input video '{input_video}' not found")
    elif not os.path.exists(weights):
        print(f"❌ Error: Model weights '{weights}' not found")
    else:
        detect_stampede_video(input_video, weights, output_video)
