import os
import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
from typing import List, Dict, Tuple, Any
import math

class GroundTruthAugmentator:
    """
    Enhanced Comprehensive image and ground truth augmentation toolkit
    Supports: YOLO, COCO JSON, Pascal VOC XML, and segmentation masks
    """
    
    def __init__(self, input_dir, output_dir, annotation_format='yolo', augmentation_factor=5, debug=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.annotation_format = annotation_format.lower()
        self.augmentation_factor = augmentation_factor
        self.debug = debug
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        if annotation_format == 'segmentation':
            (self.output_dir / 'masks').mkdir(exist_ok=True)
        
        # Supported formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_gt': 0,
            'images_without_gt': 0,
            'augmented_generated': 0,
            'failed_augmentations': 0
        }
        
        # Store class names for COCO format
        self.class_names = []
        self.coco_annotations = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.annotation_id = 1
        self.image_id = 1
    
    def debug_print(self, message):
        """Print debug messages if debug mode is enabled"""
        if self.debug:
            print(f"DEBUG: {message}")
    
    def analyze_dataset_structure(self):
        """Analyze and report dataset structure"""
        print("🔍 Analyzing dataset structure...")
        
        # Check for images
        image_files = self.get_image_files()
        self.stats['total_images'] = len(image_files)
        
        if not image_files:
            print("❌ No image files found!")
            return False
        
        print(f"📸 Found {len(image_files)} images")
        
        # Check directory structure
        images_dir = self.input_dir / 'images'
        labels_dir = self.input_dir / 'labels'
        masks_dir = self.input_dir / 'masks'
        
        print(f"📁 Input directory: {self.input_dir}")
        print(f"   - Images in root: {len(list(self.input_dir.glob('*.jpg'))) + len(list(self.input_dir.glob('*.png')))}")
        print(f"   - Images subdirectory exists: {images_dir.exists()}")
        if images_dir.exists():
            print(f"   - Images in subdirectory: {len(list(images_dir.glob('*.*')))}")
        print(f"   - Labels subdirectory exists: {labels_dir.exists()}")
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.*'))
            print(f"   - Label files found: {len(label_files)}")
            if label_files:
                extensions = set([f.suffix.lower() for f in label_files])
                print(f"   - Label file extensions: {extensions}")
        
        # Sample a few images to check ground truth availability
        sample_size = min(10, len(image_files))
        sample_images = random.sample(image_files, sample_size)
        
        print(f"\n🧪 Sampling {sample_size} images for ground truth check:")
        gt_found = 0
        for img_file in sample_images:
            gt = self.load_ground_truth(img_file)
            status = "✅" if gt is not None else "❌"
            print(f"   {status} {img_file.name}: {'GT found' if gt else 'No GT'}")
            if gt:
                gt_found += 1
                if self.annotation_format == 'yolo' and isinstance(gt, list):
                    print(f"      └─ {len(gt)} annotations")
        
        print(f"\n📊 Ground truth availability: {gt_found}/{sample_size} ({gt_found/sample_size*100:.1f}%)")
        
        if gt_found == 0:
            print("⚠️  WARNING: No ground truth found in sample! Check annotation format and file paths.")
            self.suggest_fixes()
            return False
        
        return True
    
    def suggest_fixes(self):
        """Suggest potential fixes for common issues"""
        print("\n💡 Potential fixes:")
        print("1. Check if annotation format is correct (--format yolo/pascal_voc/coco/segmentation)")
        print("2. Verify annotation files exist in the right location:")
        
        if self.annotation_format == 'yolo':
            print("   - YOLO: .txt files in labels/ subdirectory")
            print("   - Expected structure: input_dir/images/*.jpg + input_dir/labels/*.txt")
        elif self.annotation_format == 'pascal_voc':
            print("   - Pascal VOC: .xml files in labels/ subdirectory")
        elif self.annotation_format == 'segmentation':
            print("   - Segmentation: .png mask files in masks/ subdirectory")
        
        print("3. Check file naming - annotation files should have same name as images")
        print("4. Verify file permissions and paths")
        
        # Show actual files found
        labels_dir = self.input_dir / 'labels'
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.*'))[:5]  # Show first 5
            print(f"\n📋 Sample label files found:")
            for lf in label_files:
                print(f"   - {lf.name}")
    
    def get_image_files(self):
        """Get all image files from the input directory"""
        image_files = []
        
        # Check if input_dir already contains images directly
        for ext in self.image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        # If no images found directly, look in images subdirectory
        if not image_files:
            images_dir = self.input_dir / 'images'
            if images_dir.exists():
                for ext in self.image_extensions:
                    image_files.extend(images_dir.glob(f"*{ext}"))
                    image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)  # Sort for consistent processing
    
    def augment_dataset(self):
        """Main function to augment the entire dataset with ground truth"""
        # First analyze dataset structure
        if not self.analyze_dataset_structure():
            print("❌ Dataset analysis failed. Please check the issues above.")
            return
        
        image_files = self.get_image_files()
        
        print(f"\n🚀 Starting augmentation process...")
        print(f"📊 Configuration:")
        print(f"   - Annotation format: {self.annotation_format}")
        print(f"   - Augmentation factor: {self.augmentation_factor}x")
        print(f"   - Output directory: {self.output_dir}")
        print(f"   - Debug mode: {self.debug}")
        print("-" * 60)
        
        # Initialize COCO format if needed
        if self.annotation_format == 'coco':
            self.initialize_coco_format()
        
        # Copy original images and annotations
        print("📋 Copying original images and annotations...")
        for img_file in tqdm(image_files, desc="Copying originals"):
            self.copy_original_data(img_file)
        
        # Generate augmented images and annotations
        print("\n🎨 Generating augmented images and annotations...")
        
        images_with_gt = []
        images_without_gt = []
        
        # First pass: identify images with ground truth
        for img_file in image_files:
            gt = self.load_ground_truth(img_file)
            if gt is not None:
                images_with_gt.append(img_file)
            else:
                images_without_gt.append(img_file)
        
        self.stats['images_with_gt'] = len(images_with_gt)
        self.stats['images_without_gt'] = len(images_without_gt)
        
        print(f"📊 Images with ground truth: {len(images_with_gt)}")
        print(f"📊 Images without ground truth: {len(images_without_gt)}")
        
        if len(images_without_gt) > 0:
            print(f"⚠️  Warning: {len(images_without_gt)} images have no ground truth and will be skipped for augmentation")
            if self.debug and len(images_without_gt) <= 10:
                print("Images without GT:")
                for img in images_without_gt:
                    print(f"   - {img.name}")
        
        # Second pass: augment images with ground truth
        if images_with_gt:
            for img_file in tqdm(images_with_gt, desc="Augmenting images"):
                try:
                    # Load image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        self.debug_print(f"Could not load image: {img_file}")
                        continue
                    
                    # Load ground truth
                    ground_truth = self.load_ground_truth(img_file)
                    if ground_truth is None:
                        continue
                    
                    base_name = img_file.stem
                    ext = img_file.suffix.lower()
                    
                    # Generate augmented versions
                    for i in range(self.augmentation_factor - 1):  # -1 for original
                        try:
                            aug_image, aug_ground_truth = self.generate_single_augmentation(image, ground_truth)
                            
                            # Validate augmented ground truth
                            if self.validate_ground_truth(aug_ground_truth):
                                # Save augmented image
                                output_name = f"{base_name}_aug_{i+1}{ext}"
                                output_image_path = self.output_dir / 'images' / output_name
                                success = cv2.imwrite(str(output_image_path), aug_image)
                                
                                if success:
                                    # Save augmented ground truth
                                    self.save_ground_truth(output_name, aug_ground_truth, aug_image.shape)
                                    self.stats['augmented_generated'] += 1
                                    self.debug_print(f"Generated: {output_name}")
                                else:
                                    self.debug_print(f"Failed to save image: {output_name}")
                                    self.stats['failed_augmentations'] += 1
                            else:
                                self.debug_print(f"Invalid ground truth for {base_name}_aug_{i+1}")
                                self.stats['failed_augmentations'] += 1
                                
                        except Exception as e:
                            self.debug_print(f"Error generating augmentation {i+1} for {img_file.name}: {str(e)}")
                            self.stats['failed_augmentations'] += 1
                            continue
                            
                except Exception as e:
                    self.debug_print(f"Error processing {img_file}: {str(e)}")
                    continue
        
        # Finalize COCO format if needed
        if self.annotation_format == 'coco':
            self.finalize_coco_format()
        
        self.print_final_report()
    
    def validate_ground_truth(self, ground_truth):
        """Validate that ground truth is not empty or invalid"""
        if ground_truth is None:
            return False
        
        if self.annotation_format in ['yolo', 'pascal_voc']:
            return isinstance(ground_truth, list) and len(ground_truth) > 0
        elif self.annotation_format == 'segmentation':
            return ground_truth is not None and ground_truth.size > 0
        elif self.annotation_format == 'coco':
            return ground_truth is not None
        
        return True
    
    def print_final_report(self):
        """Print final augmentation report"""
        print(f"\n" + "="*60)
        print(f"🎉 AUGMENTATION COMPLETE!")
        print(f"="*60)
        print(f"📊 STATISTICS:")
        print(f"   • Total images processed: {self.stats['total_images']}")
        print(f"   • Images with ground truth: {self.stats['images_with_gt']}")
        print(f"   • Images without ground truth: {self.stats['images_without_gt']}")
        print(f"   • Augmented images generated: {self.stats['augmented_generated']}")
        print(f"   • Failed augmentations: {self.stats['failed_augmentations']}")
        print(f"   • Total images in output: {self.stats['total_images'] + self.stats['augmented_generated']}")
        
        if self.stats['images_with_gt'] > 0:
            success_rate = (self.stats['augmented_generated'] / (self.stats['images_with_gt'] * (self.augmentation_factor - 1))) * 100
            print(f"   • Augmentation success rate: {success_rate:.1f}%")
        
        print(f"\n📁 Output saved to: {self.output_dir}")
        
        if self.stats['augmented_generated'] == 0:
            print(f"\n⚠️  WARNING: No augmented images were generated!")
            print(f"   This usually means:")
            print(f"   1. No ground truth files were found")
            print(f"   2. Ground truth format doesn't match --format parameter")
            print(f"   3. File naming mismatch between images and annotations")
            print(f"   Run with --debug flag for more detailed information")

    # [Keep all the existing methods from the original class - load_ground_truth, generate_single_augmentation, etc.]
    # I'll include the key methods that might need debugging enhancements:
    
    def load_ground_truth(self, img_file):
        """Load ground truth based on annotation format with enhanced debugging"""
        base_name = img_file.stem
        
        try:
            if self.annotation_format == 'yolo':
                return self.load_yolo_annotation(base_name, img_file.parent)
            elif self.annotation_format == 'coco':
                return self.load_coco_annotation(base_name, img_file.parent)
            elif self.annotation_format == 'pascal_voc':
                return self.load_pascal_voc_annotation(base_name, img_file.parent)
            elif self.annotation_format == 'segmentation':
                return self.load_segmentation_mask(base_name, img_file.parent)
            else:
                raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
        except Exception as e:
            self.debug_print(f"Error loading ground truth for {img_file.name}: {str(e)}")
            return None
    
    def load_yolo_annotation(self, base_name, img_parent_dir):
        """Load YOLO format annotation with enhanced path searching"""
        possible_paths = []
        
        # Try multiple possible locations
        possible_paths.append(self.input_dir / 'labels' / f"{base_name}.txt")
        
        if img_parent_dir.name == 'images':
            possible_paths.append(img_parent_dir.parent / 'labels' / f"{base_name}.txt")
        
        possible_paths.append(img_parent_dir / f"{base_name}.txt")
        possible_paths.append(self.input_dir / f"{base_name}.txt")
        
        self.debug_print(f"Searching for YOLO annotation: {base_name}.txt")
        
        for label_file in possible_paths:
            self.debug_print(f"  Trying: {label_file}")
            if label_file.exists():
                self.debug_print(f"  ✅ Found: {label_file}")
                try:
                    annotations = []
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                annotations.append([class_id, x_center, y_center, width, height])
                            else:
                                self.debug_print(f"  ⚠️  Invalid YOLO format at line {line_num}: {line}")
                    
                    self.debug_print(f"  📊 Loaded {len(annotations)} annotations")
                    return annotations if annotations else None
                    
                except Exception as e:
                    self.debug_print(f"  ❌ Error reading {label_file}: {str(e)}")
                    continue
        
        self.debug_print(f"  ❌ No YOLO annotation found for {base_name}")
        return None

    def copy_original_data(self, img_file):
        """Copy original image and its annotation"""
        # Copy image
        output_image_path = self.output_dir / 'images' / f"original_{img_file.name}"
        shutil.copy2(img_file, output_image_path)
        
        # Copy annotation
        ground_truth = self.load_ground_truth(img_file)
        if ground_truth is not None:
            image = cv2.imread(str(img_file))
            self.save_ground_truth(f"original_{img_file.name}", ground_truth, image.shape)

    def load_coco_annotation(self, base_name, img_parent_dir):
        """Load COCO format annotation from JSON file"""
        possible_paths = [
            self.input_dir / 'labels' / f"{base_name}.json",
            img_parent_dir.parent / 'labels' / f"{base_name}.json" if img_parent_dir.name == 'images' else None,
            img_parent_dir / f"{base_name}.json"
        ]
        
        for json_file in filter(None, possible_paths):
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    self.debug_print(f"Error reading COCO annotation {json_file}: {str(e)}")
                    continue
        return None
    
    def load_pascal_voc_annotation(self, base_name, img_parent_dir):
        """Load Pascal VOC XML annotation"""
        possible_paths = [
            self.input_dir / 'labels' / f"{base_name}.xml",
            img_parent_dir.parent / 'labels' / f"{base_name}.xml" if img_parent_dir.name == 'images' else None,
            img_parent_dir / f"{base_name}.xml"
        ]
        
        for xml_file in filter(None, possible_paths):
            if xml_file.exists():
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    annotations = []
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        annotations.append({
                            'class_name': class_name,
                            'bbox': [xmin, ymin, xmax, ymax]
                        })
                    
                    return annotations if annotations else None
                except Exception as e:
                    self.debug_print(f"Error reading Pascal VOC annotation {xml_file}: {str(e)}")
                    continue
        return None
    
    def load_segmentation_mask(self, base_name, img_parent_dir):
        """Load segmentation mask"""
        possible_paths = [
            self.input_dir / 'masks' / f"{base_name}.png",
            img_parent_dir.parent / 'masks' / f"{base_name}.png" if img_parent_dir.name == 'images' else None,
            img_parent_dir / f"{base_name}.png"
        ]
        
        for mask_file in filter(None, possible_paths):
            if mask_file.exists():
                try:
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    return mask
                except Exception as e:
                    self.debug_print(f"Error reading segmentation mask {mask_file}: {str(e)}")
                    continue
        return None
    
    def generate_single_augmentation(self, image, ground_truth):
        """Generate a single augmented image and its corresponding ground truth"""
        # Define augmentation techniques that preserve spatial relationships
        geometric_techniques = [
            ('horizontal_flip', self.horizontal_flip),
            ('vertical_flip', self.vertical_flip),
            ('rotate_image', self.rotate_image),
            ('random_crop', self.random_crop),
            ('zoom_in', self.zoom_in),
            ('perspective_transform', self.perspective_transform),
        ]
        
        # Non-geometric techniques (don't affect ground truth)
        non_geometric_techniques = [
            ('brightness_adjustment', self.brightness_adjustment),
            ('contrast_adjustment', self.contrast_adjustment),
            ('saturation_adjustment', self.saturation_adjustment),
            ('gaussian_blur', self.gaussian_blur),
            ('noise_addition', self.noise_addition),
        ]
        
        # Select one geometric transformation
        geom_name, geom_func = random.choice(geometric_techniques)
        
        # Apply geometric transformation
        aug_image, aug_ground_truth = geom_func(image, ground_truth)
        
        # Optionally apply non-geometric transformations
        if random.random() < 0.7:  # 70% chance to apply color/filter augmentation
            non_geom_name, non_geom_func = random.choice(non_geometric_techniques)
            aug_image = non_geom_func(aug_image)
        
        return aug_image, aug_ground_truth
    
    # GEOMETRIC TRANSFORMATIONS (affect ground truth)
    def horizontal_flip(self, image, ground_truth):
        """Horizontal flip with ground truth transformation"""
        aug_image = cv2.flip(image, 1)
        
        if self.annotation_format == 'yolo':
            aug_gt = []
            for annotation in ground_truth:
                class_id, x_center, y_center, width, height = annotation
                # Flip x coordinate
                new_x_center = 1.0 - x_center
                aug_gt.append([class_id, new_x_center, y_center, width, height])
            return aug_image, aug_gt
            
        elif self.annotation_format == 'pascal_voc':
            height, width = image.shape[:2]
            aug_gt = []
            for annotation in ground_truth:
                class_name = annotation['class_name']
                xmin, ymin, xmax, ymax = annotation['bbox']
                # Flip x coordinates
                new_xmin = width - xmax
                new_xmax = width - xmin
                aug_gt.append({
                    'class_name': class_name,
                    'bbox': [new_xmin, ymin, new_xmax, ymax]
                })
            return aug_image, aug_gt
            
        elif self.annotation_format == 'segmentation':
            aug_gt = cv2.flip(ground_truth, 1)
            return aug_image, aug_gt
            
        return aug_image, ground_truth
    
    def vertical_flip(self, image, ground_truth):
        """Vertical flip with ground truth transformation"""
        aug_image = cv2.flip(image, 0)
        
        if self.annotation_format == 'yolo':
            aug_gt = []
            for annotation in ground_truth:
                class_id, x_center, y_center, width, height = annotation
                # Flip y coordinate
                new_y_center = 1.0 - y_center
                aug_gt.append([class_id, x_center, new_y_center, width, height])
            return aug_image, aug_gt
            
        elif self.annotation_format == 'pascal_voc':
            img_height, width = image.shape[:2]
            aug_gt = []
            for annotation in ground_truth:
                class_name = annotation['class_name']
                xmin, ymin, xmax, ymax = annotation['bbox']
                # Flip y coordinates
                new_ymin = img_height - ymax
                new_ymax = img_height - ymin
                aug_gt.append({
                    'class_name': class_name,
                    'bbox': [xmin, new_ymin, xmax, new_ymax]
                })
            return aug_image, aug_gt
            
        elif self.annotation_format == 'segmentation':
            aug_gt = cv2.flip(ground_truth, 0)
            return aug_image, aug_gt
            
        return aug_image, ground_truth
    
    def rotate_image(self, image, ground_truth):
        """Rotation with ground truth transformation"""
        angle = random.uniform(-15, 15)  # Smaller angles to preserve more annotations
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        aug_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        if self.annotation_format == 'yolo':
            aug_gt = []
            for annotation in ground_truth:
                class_id, x_center, y_center, w, h = annotation
                
                # Convert to pixel coordinates
                x_pixel = x_center * width
                y_pixel = y_center * height
                
                # Apply rotation to center point
                cos_angle = math.cos(math.radians(angle))
                sin_angle = math.sin(math.radians(angle))
                
                # Translate to origin, rotate, translate back
                x_centered = x_pixel - center[0]
                y_centered = y_pixel - center[1]
                
                new_x = x_centered * cos_angle - y_centered * sin_angle + center[0]
                new_y = x_centered * sin_angle + y_centered * cos_angle + center[1]
                
                # Convert back to normalized coordinates
                new_x_center = new_x / width
                new_y_center = new_y / height
                
                # Keep original width and height (approximation)
                if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1:
                    aug_gt.append([class_id, new_x_center, new_y_center, w, h])
            
            return aug_image, aug_gt
            
        elif self.annotation_format == 'segmentation':
            aug_gt = cv2.warpAffine(ground_truth, rotation_matrix, (width, height))
            return aug_image, aug_gt
            
        return aug_image, ground_truth
    
    # Completing the random_crop method and adding all remaining methods

def random_crop(self, image, ground_truth):
    """Random crop with ground truth transformation"""
    height, width = image.shape[:2]
    
    # Random crop parameters
    crop_ratio = random.uniform(0.8, 0.95)
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)
    
    start_y = random.randint(0, height - crop_height)
    start_x = random.randint(0, width - crop_width)
    
    # Crop image
    cropped = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    aug_image = cv2.resize(cropped, (width, height))
    
    if self.annotation_format == 'yolo':
        aug_gt = []
        for annotation in ground_truth:
            class_id, x_center, y_center, w, h = annotation
            
            # Convert to pixel coordinates in original image
            x_pixel = x_center * width
            y_pixel = y_center * height
            w_pixel = w * width
            h_pixel = h * height
            
            # Check if annotation overlaps with crop region
            x_min = x_pixel - w_pixel / 2
            x_max = x_pixel + w_pixel / 2
            y_min = y_pixel - h_pixel / 2
            y_max = y_pixel + h_pixel / 2
            
            # Calculate intersection with crop region
            crop_x_min = max(x_min, start_x)
            crop_x_max = min(x_max, start_x + crop_width)
            crop_y_min = max(y_min, start_y)
            crop_y_max = min(y_max, start_y + crop_height)
            
            # Check if there's significant overlap (at least 50% of original box)
            if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
                intersection_area = (crop_x_max - crop_x_min) * (crop_y_max - crop_y_min)
                original_area = w_pixel * h_pixel
                
                if intersection_area / original_area >= 0.5:
                    # Adjust coordinates relative to crop
                    new_x_center = ((crop_x_min + crop_x_max) / 2 - start_x) / crop_width
                    new_y_center = ((crop_y_min + crop_y_max) / 2 - start_y) / crop_height
                    new_width = (crop_x_max - crop_x_min) / crop_width
                    new_height = (crop_y_max - crop_y_min) / crop_height
                    
                    # Ensure coordinates are within bounds
                    if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1:
                        aug_gt.append([class_id, new_x_center, new_y_center, new_width, new_height])
        
        return aug_image, aug_gt
        
    elif self.annotation_format == 'pascal_voc':
        aug_gt = []
        for annotation in ground_truth:
            class_name = annotation['class_name']
            xmin, ymin, xmax, ymax = annotation['bbox']
            
            # Check intersection with crop region
            crop_xmin = max(xmin, start_x)
            crop_xmax = min(xmax, start_x + crop_width)
            crop_ymin = max(ymin, start_y)
            crop_ymax = min(ymax, start_y + crop_height)
            
            if crop_xmax > crop_xmin and crop_ymax > crop_ymin:
                # Calculate intersection area
                intersection_area = (crop_xmax - crop_xmin) * (crop_ymax - crop_ymin)
                original_area = (xmax - xmin) * (ymax - ymin)
                
                if intersection_area / original_area >= 0.5:
                    # Adjust coordinates relative to crop and scale
                    scale_x = width / crop_width
                    scale_y = height / crop_height
                    
                    new_xmin = int((crop_xmin - start_x) * scale_x)
                    new_ymin = int((crop_ymin - start_y) * scale_y)
                    new_xmax = int((crop_xmax - start_x) * scale_x)
                    new_ymax = int((crop_ymax - start_y) * scale_y)
                    
                    aug_gt.append({
                        'class_name': class_name,
                        'bbox': [new_xmin, new_ymin, new_xmax, new_ymax]
                    })
        
        return aug_image, aug_gt
        
    elif self.annotation_format == 'segmentation':
        cropped_mask = ground_truth[start_y:start_y+crop_height, start_x:start_x+crop_width]
        aug_gt = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return aug_image, aug_gt
        
    return aug_image, ground_truth

def zoom_in(self, image, ground_truth):
    """Zoom in with ground truth transformation"""
    height, width = image.shape[:2]
    
    # Random zoom parameters
    zoom_factor = random.uniform(1.1, 1.3)
    
    # Calculate new dimensions
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)
    
    # Resize image
    zoomed = cv2.resize(image, (new_width, new_height))
    
    # Calculate crop coordinates to get back to original size
    start_y = (new_height - height) // 2
    start_x = (new_width - width) // 2
    
    # Crop to original size
    aug_image = zoomed[start_y:start_y+height, start_x:start_x+width]
    
    if self.annotation_format == 'yolo':
        aug_gt = []
        for annotation in ground_truth:
            class_id, x_center, y_center, w, h = annotation
            
            # Scale coordinates
            new_x_center = x_center * zoom_factor - (start_x / width)
            new_y_center = y_center * zoom_factor - (start_y / height)
            new_width = w * zoom_factor
            new_height = h * zoom_factor
            
            # Check if annotation is still within bounds
            if (0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and
                new_width > 0 and new_height > 0):
                aug_gt.append([class_id, new_x_center, new_y_center, new_width, new_height])
        
        return aug_image, aug_gt
        
    elif self.annotation_format == 'segmentation':
        zoomed_mask = cv2.resize(ground_truth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        aug_gt = zoomed_mask[start_y:start_y+height, start_x:start_x+width]
        return aug_image, aug_gt
        
    return aug_image, ground_truth

def perspective_transform(self, image, ground_truth):
    """Perspective transformation with ground truth transformation"""
    height, width = image.shape[:2]
    
    # Define perspective transformation points
    offset = random.randint(10, 30)
    
    # Source points (corners of original image)
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    # Destination points (slightly skewed)
    dst_points = np.float32([
        [random.randint(0, offset), random.randint(0, offset)],
        [width - random.randint(0, offset), random.randint(0, offset)],
        [width - random.randint(0, offset), height - random.randint(0, offset)],
        [random.randint(0, offset), height - random.randint(0, offset)]
    ])
    
    # Get transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    aug_image = cv2.warpPerspective(image, transform_matrix, (width, height))
    
    if self.annotation_format == 'yolo':
        aug_gt = []
        for annotation in ground_truth:
            class_id, x_center, y_center, w, h = annotation
            
            # Convert to corner coordinates
            x_pixel = x_center * width
            y_pixel = y_center * height
            w_pixel = w * width
            h_pixel = h * height
            
            # Get bounding box corners
            corners = np.float32([
                [x_pixel - w_pixel/2, y_pixel - h_pixel/2],
                [x_pixel + w_pixel/2, y_pixel - h_pixel/2],
                [x_pixel + w_pixel/2, y_pixel + h_pixel/2],
                [x_pixel - w_pixel/2, y_pixel + h_pixel/2]
            ]).reshape(-1, 1, 2)
            
            # Transform corners
            transformed_corners = cv2.perspectiveTransform(corners, transform_matrix)
            transformed_corners = transformed_corners.reshape(-1, 2)
            
            # Calculate new bounding box
            x_coords = transformed_corners[:, 0]
            y_coords = transformed_corners[:, 1]
            
            new_xmin = np.min(x_coords)
            new_xmax = np.max(x_coords)
            new_ymin = np.min(y_coords)
            new_ymax = np.max(y_coords)
            
            # Convert back to YOLO format
            new_x_center = (new_xmin + new_xmax) / (2 * width)
            new_y_center = (new_ymin + new_ymax) / (2 * height)
            new_width = (new_xmax - new_xmin) / width
            new_height = (new_ymax - new_ymin) / height
            
            # Check if annotation is still within bounds
            if (0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and
                new_width > 0 and new_height > 0):
                aug_gt.append([class_id, new_x_center, new_y_center, new_width, new_height])
        
        return aug_image, aug_gt
        
    elif self.annotation_format == 'segmentation':
        aug_gt = cv2.warpPerspective(ground_truth, transform_matrix, (width, height))
        return aug_image, aug_gt
        
    return aug_image, ground_truth

# NON-GEOMETRIC TRANSFORMATIONS (don't affect ground truth)
def brightness_adjustment(self, image):
    """Adjust brightness"""
    factor = random.uniform(0.7, 1.3)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

def contrast_adjustment(self, image):
    """Adjust contrast"""
    factor = random.uniform(0.8, 1.2)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

def saturation_adjustment(self, image):
    """Adjust saturation"""
    factor = random.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def gaussian_blur(self, image):
    """Apply Gaussian blur"""
    kernel_size = random.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def noise_addition(self, image):
    """Add random noise"""
    noise = np.random.randint(-20, 21, image.shape, dtype=np.int16)
    noisy_image = image.astype(np.int16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def save_ground_truth(self, image_name, ground_truth, image_shape):
    """Save ground truth in the appropriate format"""
    base_name = Path(image_name).stem
    
    if self.annotation_format == 'yolo':
        self.save_yolo_annotation(base_name, ground_truth)
    elif self.annotation_format == 'pascal_voc':
        self.save_pascal_voc_annotation(base_name, ground_truth, image_shape)
    elif self.annotation_format == 'coco':
        self.save_coco_annotation(base_name, ground_truth, image_shape)
    elif self.annotation_format == 'segmentation':
        self.save_segmentation_mask(base_name, ground_truth)

def save_yolo_annotation(self, base_name, annotations):
    """Save YOLO format annotation"""
    output_file = self.output_dir / 'labels' / f"{base_name}.txt"
    with open(output_file, 'w') as f:
        for annotation in annotations:
            class_id, x_center, y_center, width, height = annotation
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def save_pascal_voc_annotation(self, base_name, annotations, image_shape):
    """Save Pascal VOC XML annotation"""
    height, width, channels = image_shape
    
    root = ET.Element("annotation")
    
    # Add basic info
    ET.SubElement(root, "filename").text = f"{base_name}.jpg"
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(channels)
    
    # Add objects
    for annotation in annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = annotation['class_name']
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = annotation['bbox']
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))
    
    # Write to file
    tree = ET.ElementTree(root)
    output_file = self.output_dir / 'labels' / f"{base_name}.xml"
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def save_segmentation_mask(self, base_name, mask):
    """Save segmentation mask"""
    output_file = self.output_dir / 'masks' / f"{base_name}.png"
    cv2.imwrite(str(output_file), mask)

def initialize_coco_format(self):
    """Initialize COCO format structure"""
    # This would be implemented based on the existing COCO data structure
    pass

def save_coco_annotation(self, base_name, annotation, image_shape):
    """Save COCO format annotation"""
    # This would be implemented to add to the COCO structure
    pass

def finalize_coco_format(self):
    """Finalize and save COCO format JSON"""
    output_file = self.output_dir / 'labels' / 'annotations.json'
    with open(output_file, 'w') as f:
        json.dump(self.coco_annotations, f, indent=2)


def main():
    """Main function to run the augmentator"""
    parser = argparse.ArgumentParser(description='Enhanced Ground Truth Augmentator')
    parser.add_argument('input_dir', help='Input directory containing images and annotations')
    parser.add_argument('output_dir', help='Output directory for augmented dataset')
    parser.add_argument('--format', choices=['yolo', 'pascal_voc', 'coco', 'segmentation'], 
                       default='yolo', help='Annotation format (default: yolo)')
    parser.add_argument('--factor', type=int, default=5, 
                       help='Augmentation factor (default: 5)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode for detailed logging')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Ground Truth Augmentator")
    print("="*50)
    
    # Create augmentator instance
    augmentator = GroundTruthAugmentator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotation_format=args.format,
        augmentation_factor=args.factor,
        debug=args.debug
    )
    
    # Run augmentation
    augmentator.augment_dataset()


if __name__ == "__main__":
    main()