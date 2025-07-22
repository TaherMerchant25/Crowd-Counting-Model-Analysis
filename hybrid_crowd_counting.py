# Hybrid Semantic Segmentation + Detectron2 Crowd Counting
# Optimized for Google Colab

import os
import sys
import numpy as np
import scipy.io as sio
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Google Colab specific imports and setup
try:
    from google.colab import drive, files
    IN_COLAB = True
    print("Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally")

# Install required packages for Colab
def install_dependencies():
    """Install required packages for Google Colab"""
    if IN_COLAB:
        print("Installing dependencies for Google Colab...")
        os.system("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html")
        os.system("pip install segmentation-models-pytorch")
        os.system("pip install albumentations")
        os.system("pip install timm")
        print("Dependencies installed!")

# Mount Google Drive if in Colab
def setup_colab_environment():
    """Setup Google Colab environment"""
    if IN_COLAB:
        drive.mount('/content/drive')
        print("Google Drive mounted!")
        return True
    return False

# Detectron2 imports (will install if not available)
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    print("Detectron2 not found. Will install in Colab.")
    DETECTRON2_AVAILABLE = False

# Segmentation model imports
try:
    import segmentation_models_pytorch as smp
    import albumentations as A
    SMP_AVAILABLE = True
except ImportError:
    print("Segmentation models not found. Will install in Colab.")
    SMP_AVAILABLE = False

# Dataset Configuration
class Config:
    """Configuration class for easy parameter adjustment"""
    def __init__(self):
        # Paths (adjust for your Google Drive structure)
        if IN_COLAB:
            self.base_path = "/content/drive/MyDrive/crowd_wala_dataset/crowd_wala_dataset"
        else:
            self.base_path = "d:/Downloads/Code_Autonomous/Project_AIMS/crowd_wala_dataset"
        
        self.train_image_dir = os.path.join(self.base_path, "train_data/images")
        self.train_gt_dir = os.path.join(self.base_path, "train_data/ground_truth")
        self.test_image_dir = os.path.join(self.base_path, "test_data/images")
        self.test_gt_dir = os.path.join(self.base_path, "test_data/ground_truth")
        
        # Model parameters
        self.img_size = (512, 512)
        self.downsample = 8
        self.batch_size = 4
        self.num_epochs = 50
        self.learning_rate = 1e-4
        
        # Segmentation classes
        self.num_classes = 5  # background, low, medium, high, extreme density
        self.density_thresholds = [0, 5, 15, 30, 50]  # people per region
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

config = Config()

# Utility Functions
def load_gt_from_mat(mat_file_path, image_shape):
    """Load ground truth points from .mat file"""
    try:
        mat = sio.loadmat(mat_file_path)
        ann_points = mat['image_info'][0][0][0][0][0]
        return ann_points
    except:
        try:
            mat = sio.loadmat(mat_file_path)
            # Try alternative structure
            ann_points = mat['image_info'][0][0][0]
            return ann_points
        except:
            print(f"Error loading {mat_file_path}")
            return np.array([])

def create_density_map(points, image_shape, sigma=15):
    """Create Gaussian density map from point annotations"""
    density_map = np.zeros(image_shape[:2], dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    for point in points:
        x = min(int(point[0]), image_shape[1] - 1)
        y = min(int(point[1]), image_shape[0] - 1)
        density_map[y, x] += 1
    
    # Apply Gaussian blur
    density_map = cv2.GaussianBlur(density_map, (sigma, sigma), 0)
    return density_map

def create_segmentation_mask(points, image_shape, region_size=32):
    """Create semantic segmentation mask from density"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if len(points) == 0:
        return mask
    
    # Create grid-based density classification
    h, w = image_shape[:2]
    for y in range(0, h, region_size):
        for x in range(0, w, region_size):
            # Count points in this region
            region_points = 0
            for point in points:
                px, py = int(point[0]), int(point[1])
                if x <= px < x + region_size and y <= py < y + region_size:
                    region_points += 1
            
            # Classify density level
            density_class = 0
            for i, threshold in enumerate(config.density_thresholds[1:], 1):
                if region_points >= threshold:
                    density_class = i
            
            # Fill region with class
            mask[y:y+region_size, x:x+region_size] = density_class
    
    return mask

# Dataset Classes
class HybridCrowdDataset(Dataset):
    """Dataset for hybrid crowd counting with segmentation and detection"""
    
    def __init__(self, image_dir, gt_dir, transform=None, mode='train'):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.mode = mode
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Found {len(self.image_files)} images in {mode} set")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth
        gt_name = f"GT_{img_name.split('.')[0]}.mat"
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        points = load_gt_from_mat(gt_path, image.shape)
        
        # Create different ground truth formats
        density_map = create_density_map(points, image.shape)
        seg_mask = create_segmentation_mask(points, image.shape)
        count = len(points)
        
        # Resize to target size
        image = cv2.resize(image, config.img_size)
        density_map = cv2.resize(density_map, 
                               (config.img_size[0]//config.downsample, 
                                config.img_size[1]//config.downsample))
        seg_mask = cv2.resize(seg_mask, config.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=seg_mask)
            image = transformed['image']
            seg_mask = transformed['mask']
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        density_map = torch.from_numpy(density_map).float()
        seg_mask = torch.from_numpy(seg_mask).long()
        
        return {
            'image': image,
            'density_map': density_map,
            'seg_mask': seg_mask,
            'count': torch.tensor(count, dtype=torch.float32),
            'points': torch.from_numpy(points).float() if len(points) > 0 else torch.zeros((0, 2))
        }

# Detectron2 Integration
class Detectron2PersonDetector:
    """Wrapper for Detectron2 person detection"""
    
    def __init__(self):
        if not DETECTRON2_AVAILABLE:
            print("Detectron2 not available. Install it first.")
            return
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
    
    def detect_persons(self, image):
        """Detect persons in image and return bounding boxes"""
        if not DETECTRON2_AVAILABLE:
            return []
        
        outputs = self.predictor(image)
        instances = outputs["instances"]
        
        # Filter for person class (class 0 in COCO)
        person_mask = instances.pred_classes == 0
        person_boxes = instances.pred_boxes[person_mask].tensor.cpu().numpy()
        person_scores = instances.scores[person_mask].cpu().numpy()
        
        return person_boxes, person_scores

# Hybrid Neural Network Architecture
class AttentionBlock(nn.Module):
    """Attention mechanism for feature fusion"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction backbone"""
    
    def __init__(self, backbone='resnet50'):
        super().__init__()
        
        if SMP_AVAILABLE:
            # Use segmentation models pytorch for better feature extraction
            self.encoder = smp.encoders.get_encoder(
                backbone, 
                in_channels=3, 
                depth=5, 
                weights='imagenet'
            )
        else:
            # Fallback to basic CNN
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        
        # Attention modules
        self.attention1 = AttentionBlock(256)
        self.attention2 = AttentionBlock(512)
    
    def forward(self, x):
        if SMP_AVAILABLE:
            features = self.encoder(x)
            return features
        else:
            x = self.encoder(x)
            return [x]

class SegmentationHead(nn.Module):
    """Semantic segmentation head"""
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, x):
        return self.decoder(x)

class DensityHead(nn.Module):
    """Density estimation head"""
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.decoder(x)

class HybridCrowdModel(nn.Module):
    """Hybrid model combining segmentation and density estimation"""
    
    def __init__(self, num_classes=5, backbone='resnet50'):
        super().__init__()
        
        self.backbone = MultiScaleFeatureExtractor(backbone)
        
        # Determine feature dimensions
        if SMP_AVAILABLE:
            encoder_channels = [3, 64, 256, 512, 1024, 2048]  # ResNet50 channels
            feature_channels = encoder_channels[-1]
        else:
            feature_channels = 512
        
        # Multiple heads
        self.seg_head = SegmentationHead(feature_channels, num_classes)
        self.density_head = DensityHead(feature_channels)
        
        # Feature fusion
        self.fusion = nn.Conv2d(feature_channels * 2, feature_channels, 1)
        
        # Upsampling for final outputs
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        if isinstance(features, list):
            main_features = features[-1]
        else:
            main_features = features
        
        # Generate outputs
        seg_logits = self.seg_head(main_features)
        density_map = self.density_head(main_features)
        
        # Upsample to original size
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        density_map = F.interpolate(density_map, size=(x.shape[-2:]//config.downsample), 
                                  mode='bilinear', align_corners=False)
        
        return {
            'segmentation': seg_logits,
            'density': density_map,
            'features': main_features
        }

# Loss Functions
class HybridLoss(nn.Module):
    """Combined loss for hybrid training"""
    
    def __init__(self, seg_weight=1.0, density_weight=1.0, count_weight=1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.density_weight = density_weight
        self.count_weight = count_weight
        
        self.seg_loss = nn.CrossEntropyLoss()
        self.density_loss = nn.MSELoss()
        self.count_loss = nn.L1Loss()
    
    def forward(self, outputs, targets):
        seg_logits = outputs['segmentation']
        density_pred = outputs['density']
        
        seg_target = targets['seg_mask']
        density_target = targets['density_map']
        count_target = targets['count']
        
        # Segmentation loss
        loss_seg = self.seg_loss(seg_logits, seg_target)
        
        # Density loss
        loss_density = self.density_loss(density_pred.squeeze(1), density_target)
        
        # Count loss
        count_pred = density_pred.sum(dim=(2, 3)).squeeze(1)
        loss_count = self.count_loss(count_pred, count_target)
        
        # Total loss
        total_loss = (self.seg_weight * loss_seg + 
                     self.density_weight * loss_density + 
                     self.count_weight * loss_count)
        
        return {
            'total_loss': total_loss,
            'seg_loss': loss_seg,
            'density_loss': loss_density,
            'count_loss': loss_count
        }

# Training Functions
def get_transforms():
    """Get augmentation transforms"""
    if 'albumentations' in sys.modules:
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
        ])
        return train_transform
    else:
        return None

def train_model(model, train_loader, val_loader, num_epochs=50):
    """Training loop for hybrid model"""
    
    criterion = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    model.to(config.device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move to device
            images = batch['image'].to(config.device)
            targets = {
                'seg_mask': batch['seg_mask'].to(config.device),
                'density_map': batch['density_map'].to(config.device),
                'count': batch['count'].to(config.device)
            }
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            losses = criterion(outputs, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += losses['total_loss'].item()
            
            # Count accuracy
            pred_count = outputs['density'].sum(dim=(2, 3)).squeeze(1)
            mae = torch.abs(pred_count - targets['count']).mean()
            train_mae += mae.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(config.device)
                targets = {
                    'seg_mask': batch['seg_mask'].to(config.device),
                    'density_map': batch['density_map'].to(config.device),
                    'count': batch['count'].to(config.device)
                }
                
                outputs = model(images)
                losses = criterion(outputs, targets)
                
                val_loss += losses['total_loss'].item()
                
                pred_count = outputs['density'].sum(dim=(2, 3)).squeeze(1)
                mae = torch.abs(pred_count - targets['count']).mean()
                val_mae += mae.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.2f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.2f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), 'best_hybrid_model.pth')
            print(f'  New best model saved!')
    
    return train_losses, val_losses

# Inference and Visualization
class HybridInference:
    """Inference class for hybrid model"""
    
    def __init__(self, model_path, use_detectron2=True):
        self.model = HybridCrowdModel()
        self.model.load_state_dict(torch.load(model_path, map_location=config.device))
        self.model.to(config.device)
        self.model.eval()
        
        # Initialize Detectron2 if available
        self.detectron2 = None
        if use_detectron2 and DETECTRON2_AVAILABLE:
            self.detectron2 = Detectron2PersonDetector()
    
    def predict(self, image_path):
        """Run inference on single image"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape
        
        # Resize for model
        image_resized = cv2.resize(image_rgb, config.img_size)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            # Model prediction
            outputs = self.model(image_tensor)
            
            # Get predictions
            seg_pred = torch.argmax(outputs['segmentation'], dim=1).cpu().numpy()[0]
            density_pred = outputs['density'].cpu().numpy()[0, 0]
            
            # Resize back to original
            seg_pred = cv2.resize(seg_pred.astype(np.uint8), 
                                (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Count from density map
            model_count = float(outputs['density'].sum())
            
            # Detectron2 prediction if available
            detectron2_count = 0
            person_boxes = []
            if self.detectron2:
                boxes, scores = self.detectron2.detect_persons(image)
                detectron2_count = len(boxes)
                person_boxes = boxes
        
        return {
            'segmentation': seg_pred,
            'density_map': density_pred,
            'model_count': model_count,
            'detectron2_count': detectron2_count,
            'person_boxes': person_boxes,
            'hybrid_count': self._combine_counts(model_count, detectron2_count, seg_pred)
        }
    
    def _combine_counts(self, model_count, detectron2_count, seg_mask):
        """Intelligently combine model and detectron2 counts"""
        # Calculate crowd density from segmentation
        total_pixels = seg_mask.size
        crowd_pixels = np.sum(seg_mask > 0)
        crowd_ratio = crowd_pixels / total_pixels
        
        # Adaptive weighting based on crowd density
        if crowd_ratio < 0.1:  # Sparse crowd - trust detectron2 more
            weight_detectron2 = 0.7
        elif crowd_ratio > 0.5:  # Dense crowd - trust model more
            weight_detectron2 = 0.2
        else:  # Medium density - balanced approach
            weight_detectron2 = 0.5
        
        weight_model = 1 - weight_detectron2
        
        hybrid_count = weight_model * model_count + weight_detectron2 * detectron2_count
        return hybrid_count

def visualize_results(image_path, results, save_path=None):
    """Visualize all prediction results"""
    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Segmentation result
    seg_colored = plt.cm.viridis(results['segmentation'] / config.num_classes)
    axes[0, 1].imshow(seg_colored)
    axes[0, 1].set_title('Crowd Segmentation')
    axes[0, 1].axis('off')
    
    # Density map
    axes[0, 2].imshow(results['density_map'], cmap='hot')
    axes[0, 2].set_title('Density Map')
    axes[0, 2].axis('off')
    
    # Detectron2 detections
    img_with_boxes = image_rgb.copy()
    for box in results['person_boxes']:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axes[1, 0].imshow(img_with_boxes)
    axes[1, 0].set_title(f'Detectron2 Detections: {results["detectron2_count"]}')
    axes[1, 0].axis('off')
    
    # Combined visualization
    combined = image_rgb.copy()
    # Overlay segmentation
    seg_overlay = plt.cm.viridis(results['segmentation'] / config.num_classes)[:, :, :3]
    combined = cv2.addWeighted(combined.astype(np.float32), 0.7, 
                             (seg_overlay * 255).astype(np.float32), 0.3, 0)
    # Add bounding boxes
    for box in results['person_boxes']:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    axes[1, 1].imshow(combined.astype(np.uint8))
    axes[1, 1].set_title('Combined Result')
    axes[1, 1].axis('off')
    
    # Count comparison
    counts = ['Model', 'Detectron2', 'Hybrid']
    values = [results['model_count'], results['detectron2_count'], results['hybrid_count']]
    axes[1, 2].bar(counts, values)
    axes[1, 2].set_title('Count Comparison')
    axes[1, 2].set_ylabel('Count')
    
    # Add count values as text
    for i, v in enumerate(values):
        axes[1, 2].text(i, v + max(values) * 0.01, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Main execution functions
def main():
    """Main function to run the complete pipeline"""
    
    print("🚀 Starting Hybrid Crowd Counting Pipeline")
    print("=" * 50)
    
    # Setup environment
    if IN_COLAB:
        install_dependencies()
        setup_colab_environment()
    
    # Verify data paths
    if not os.path.exists(config.train_image_dir):
        print(f"❌ Training images not found at: {config.train_image_dir}")
        print("Please adjust the paths in Config class")
        return
    
    print(f"✅ Found training data at: {config.train_image_dir}")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_transform = get_transforms()
    
    train_dataset = HybridCrowdDataset(
        config.train_image_dir, 
        config.train_gt_dir, 
        transform=train_transform,
        mode='train'
    )
    
    # Split train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    print(f"✅ Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
    
    # Create model
    print("\n🤖 Creating hybrid model...")
    model = HybridCrowdModel(num_classes=config.num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Train model
    print("\n🏋️ Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, config.num_epochs)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.gradient(val_losses), label='Val Loss Gradient')
    plt.title('Learning Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Gradient')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Training completed!")
    
    # Test inference
    if os.path.exists(config.test_image_dir):
        print("\n🔍 Testing inference...")
        test_images = [f for f in os.listdir(config.test_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_images:
            # Initialize inference
            inference = HybridInference('best_hybrid_model.pth', use_detectron2=DETECTRON2_AVAILABLE)
            
            # Test on first few images
            for i, img_name in enumerate(test_images[:3]):
                img_path = os.path.join(config.test_image_dir, img_name)
                print(f"Processing: {img_name}")
                
                results = inference.predict(img_path)
                visualize_results(img_path, results)
                
                print(f"Results for {img_name}:")
                print(f"  Model Count: {results['model_count']:.1f}")
                print(f"  Detectron2 Count: {results['detectron2_count']}")
                print(f"  Hybrid Count: {results['hybrid_count']:.1f}")
                print()
    
    print("🎉 Pipeline completed successfully!")

# Quick demo function for Colab
def quick_demo():
    """Quick demonstration with sample data"""
    print("🎮 Running Quick Demo...")
    
    # Create sample data if real data not available
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    sample_points = np.random.rand(20, 2) * 512
    
    # Create model
    model = HybridCrowdModel(num_classes=5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("Model outputs:")
    print(f"  Segmentation shape: {outputs['segmentation'].shape}")
    print(f"  Density shape: {outputs['density'].shape}")
    print(f"  Predicted count: {outputs['density'].sum().item():.2f}")
    
    # Test loss
    criterion = HybridLoss()
    dummy_targets = {
        'seg_mask': torch.randint(0, 5, (1, 512, 512)),
        'density_map': torch.randn(1, 64, 64),
        'count': torch.tensor([15.0])
    }
    
    losses = criterion(outputs, dummy_targets)
    print(f"\nLoss components:")
    print(f"  Segmentation loss: {losses['seg_loss']:.4f}")
    print(f"  Density loss: {losses['density_loss']:.4f}")
    print(f"  Count loss: {losses['count_loss']:.4f}")
    print(f"  Total loss: {losses['total_loss']:.4f}")
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    # Run quick demo if no data available, otherwise run full pipeline
    if os.path.exists(config.train_image_dir):
        main()
    else:
        print("No training data found. Running quick demo instead...")
        quick_demo()
