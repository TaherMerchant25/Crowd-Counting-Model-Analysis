import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque
import time
import json
from datetime import datetime
import threading
import queue

# Import the model architecture (same as your training code)
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedConvBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SelfAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.out_conv(out)
        out = self.gamma * out + x
        return out

class DMCountPlusPlus(nn.Module):
    def __init__(self):
        super(DMCountPlusPlus, self).__init__()
        vgg = models.vgg16(pretrained=False)  # Don't need pretrained for inference
        self.frontend = nn.Sequential(*list(vgg.features.children())[:-3])
        
        self.reg_layer1 = DilatedConvBlock(512, 256, dilation=2)
        self.reg_layer2 = DilatedConvBlock(256, 128, dilation=2)
        self.ms_conv1 = DilatedConvBlock(128, 64, dilation=1)
        self.ms_conv2 = DilatedConvBlock(128, 64, dilation=2)
        self.ms_conv3 = DilatedConvBlock(128, 64, dilation=3)
        
        self.fusion = nn.Conv2d(192, 64, 1)
        self.fusion_bn = nn.BatchNorm2d(64)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.self_attention = SelfAttentionModule(64, reduction=8)
        self.density_map = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.reg_layer1(x)
        x = self.reg_layer2(x)
        
        ms1 = self.ms_conv1(x)
        ms2 = self.ms_conv2(x)
        ms3 = self.ms_conv3(x)
        ms_features = torch.cat([ms1, ms2, ms3], dim=1)
        
        fused = self.fusion_relu(self.fusion_bn(self.fusion(ms_features)))
        refined = self.self_attention(fused)
        density = self.density_map(refined)
        return density

class ZoneAnalyzer:
    """Analyze crowd density in predefined zones"""
    def __init__(self, zones_config):
        self.zones = zones_config
        self.zone_history = {zone['name']: deque(maxlen=30) for zone in zones_config}
        self.zone_alerts = {}
    
    def analyze_zones(self, density_map):
        """Analyze density in each predefined zone"""
        results = {}
        h, w = density_map.shape
        
        for zone in self.zones:
            name = zone['name']
            coords = zone['coordinates']  # [(x1,y1), (x2,y2), ...]
            threshold = zone.get('threshold', 10)
            
            # Create mask for the zone
            mask = np.zeros((h, w), dtype=bool)
            if len(coords) >= 3:  # Polygon
                points = np.array(coords, dtype=np.int32)
                cv2.fillPoly(mask, [points], True)
            else:  # Rectangle
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                mask[y1:y2, x1:x2] = True
            
            # Calculate zone density
            zone_density = np.sum(density_map * mask)
            self.zone_history[name].append(zone_density)
            
            # Calculate statistics
            avg_density = np.mean(list(self.zone_history[name])) if self.zone_history[name] else 0
            is_alert = zone_density > threshold
            
            results[name] = {
                'current_density': zone_density,
                'average_density': avg_density,
                'is_alert': is_alert,
                'threshold': threshold,
                'coords': coords
            }
            
            self.zone_alerts[name] = is_alert
        
        return results

class RealTimeCrowdHeatmap:
    def __init__(self, model_path, zones_config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = DMCountPlusPlus().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Zone analyzer
        self.zone_analyzer = ZoneAnalyzer(zones_config) if zones_config else None
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.density_history = deque(maxlen=100)
        
        # Heatmap settings
        self.heatmap_alpha = 0.6
        self.colormap = cv2.COLORMAP_JET
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model inference"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        tensor_image = self.transform(pil_image).unsqueeze(0)
        return tensor_image.to(self.device)
    
    def generate_density_map(self, frame):
        """Generate density map from frame"""
        start_time = time.time()
        
        with torch.no_grad():
            input_tensor = self.preprocess_frame(frame)
            density_output = self.model(input_tensor)
            
            # Convert to numpy
            density_map = density_output.squeeze().cpu().numpy()
            
            # Resize to original frame size
            h, w = frame.shape[:2]
            density_map_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
        # Track performance
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_counter.append(fps)
        
        return density_map_resized
    
    def create_heatmap_overlay(self, frame, density_map):
        """Create heatmap overlay on the original frame"""
        # Normalize density map for visualization
        if density_map.max() > 0:
            normalized_density = (density_map / density_map.max() * 255).astype(np.uint8)
        else:
            normalized_density = np.zeros_like(density_map, dtype=np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(normalized_density, self.colormap)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 1 - self.heatmap_alpha, heatmap, self.heatmap_alpha, 0)
        
        return overlay, heatmap
    
    def draw_zone_analysis(self, frame, zone_results):
        """Draw zone boundaries and analysis on frame"""
        if not zone_results:
            return frame
        
        overlay = frame.copy()
        
        for zone_name, data in zone_results.items():
            coords = data['coords']
            current_density = data['current_density']
            is_alert = data['is_alert']
            
            # Choose color based on alert status
            color = (0, 0, 255) if is_alert else (0, 255, 0)  # Red for alert, Green for normal
            
            # Draw zone boundary
            if len(coords) >= 3:  # Polygon
                points = np.array(coords, dtype=np.int32)
                cv2.polylines(overlay, [points], True, color, 2)
                
                # Fill with transparent color
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
                
                # Calculate centroid for text placement
                M = cv2.moments(points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = coords[0]
            else:  # Rectangle
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Fill with transparent color
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Add text information
            text = f"{zone_name}: {current_density:.1f}"
            if is_alert:
                text += " ALERT!"
            
            # Text background
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (cx - text_width//2 - 5, cy - text_height - 5),
                         (cx + text_width//2 + 5, cy + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, text, (cx - text_width//2, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_statistics(self, frame, total_count, zone_results=None):
        """Draw statistics on frame"""
        h, w = frame.shape[:2]
        
        # Current statistics
        avg_fps = np.mean(list(self.fps_counter)) if self.fps_counter else 0
        
        # Statistics panel background
        panel_height = 120 + (len(zone_results) * 25 if zone_results else 0)
        cv2.rectangle(frame, (10, 10), (300, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, panel_height), (255, 255, 255), 2)
        
        # Main statistics
        y_offset = 35
        cv2.putText(frame, f"Total Count: {total_count:.0f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Device: {str(self.device).upper()}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Zone statistics
        if zone_results:
            y_offset += 35
            cv2.putText(frame, "Zone Analysis:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for zone_name, data in zone_results.items():
                y_offset += 25
                status = "ALERT" if data['is_alert'] else "OK"
                color = (0, 0, 255) if data['is_alert'] else (0, 255, 0)
                cv2.putText(frame, f"{zone_name}: {status}", (30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def process_video_stream(self, source=0, zones_config=None, save_output=False, output_path='crowd_heatmap_output.mp4'):
        """Process video stream with real-time heatmap generation"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Video writer for saving output
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting real-time crowd density heatmap...")
        print("Press 'q' to quit, 's' to save screenshot, 'h' to toggle heatmap")
        
        show_heatmap = True
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Generate density map
                density_map = self.generate_density_map(frame)
                total_count = np.sum(density_map)
                self.density_history.append(total_count)
                
                # Zone analysis
                zone_results = None
                if self.zone_analyzer:
                    zone_results = self.zone_analyzer.analyze_zones(density_map)
                
                # Create visualization
                if show_heatmap:
                    overlay_frame, heatmap = self.create_heatmap_overlay(frame, density_map)
                else:
                    overlay_frame = frame.copy()
                
                # Draw zone analysis
                if zone_results:
                    overlay_frame = self.draw_zone_analysis(overlay_frame, zone_results)
                
                # Draw statistics
                overlay_frame = self.draw_statistics(overlay_frame, total_count, zone_results)
                
                # Display
                cv2.imshow('Crowd Density Heatmap', overlay_frame)
                
                # Save frame if required
                if save_output:
                    out.write(overlay_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"crowd_heatmap_screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, overlay_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord('h'):
                    show_heatmap = not show_heatmap
                    print(f"Heatmap display: {'ON' if show_heatmap else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            cap.release()
            if save_output:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"Processed {frame_count} frames")
            if self.fps_counter:
                print(f"Average FPS: {np.mean(list(self.fps_counter)):.2f}")

def create_sample_zones_config():
    """Create sample zone configurations for different environments"""
    
    # Retail store zones
    retail_zones = [
        {
            "name": "Entrance",
            "coordinates": [(50, 50), (200, 150)],  # Rectangle
            "threshold": 15,
            "description": "Main entrance area"
        },
        {
            "name": "Electronics",
            "coordinates": [(300, 100), (500, 250)],
            "threshold": 20,
            "description": "Electronics section"
        },
        {
            "name": "Checkout",
            "coordinates": [(100, 400), (400, 500)],
            "threshold": 25,
            "description": "Checkout counters"
        }
    ]
    
    # Museum zones
    museum_zones = [
        {
            "name": "Main_Exhibit",
            "coordinates": [(200, 150), (450, 300)],
            "threshold": 12,
            "description": "Main exhibition hall"
        },
        {
            "name": "Interactive_Zone",
            "coordinates": [(100, 350), (300, 450)],
            "threshold": 18,
            "description": "Interactive display area"
        },
        {
            "name": "Gift_Shop",
            "coordinates": [(500, 400), (650, 500)],
            "threshold": 10,
            "description": "Museum gift shop"
        }
    ]
    
    # Event space zones (polygon example)
    event_zones = [
        {
            "name": "Stage_Area",
            "coordinates": [(300, 100), (500, 100), (450, 200), (350, 200)],  # Polygon
            "threshold": 30,
            "description": "Main stage viewing area"
        },
        {
            "name": "Food_Court",
            "coordinates": [(50, 300), (250, 400)],
            "threshold": 15,
            "description": "Food and beverage area"
        },
        {
            "name": "Emergency_Exit",
            "coordinates": [(600, 50), (700, 150)],
            "threshold": 8,
            "description": "Emergency exit corridor"
        }
    ]
    
    return {
        "retail": retail_zones,
        "museum": museum_zones,
        "event": event_zones
    }

# Usage examples
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"C:\Users\taher\Downloads\best_dmcount_model.pth"  # Update this path
    
    # Create zone configurations
    zones_configs = create_sample_zones_config()
    
    print("Available zone configurations:")
    for env_type in zones_configs.keys():
        print(f"- {env_type}")
    
    # Choose environment type
    environment = "retail"  # Change to "museum" or "event" as needed
    zones_config = zones_configs[environment]
    
    # Initialize heatmap system
    heatmap_system = RealTimeCrowdHeatmap(
        model_path=MODEL_PATH,
        zones_config=zones_config
    )
    
    # Process video stream
    # Options:
    # - source=0: Default webcam
    # - source="path/to/video.mp4"
    # - source="rtsp://camera_ip/stream": IP camera
    
    heatmap_system.process_video_stream(
        source=0,  # Use webcam
        save_output=True,  # Save output video
        output_path=f"crowd_heatmap_{environment}_demo.mp4"
    )
