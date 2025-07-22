import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

class ImageAugmentator:
    """
    Comprehensive image augmentation toolkit for expanding image datasets
    Supports: mirroring, cropping, color augmentation, rotation, and filters
    """
    
    def __init__(self, input_dir, output_dir, augmentation_factor=5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.augmentation_factor = augmentation_factor
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def get_image_files(self):
        """Get all image files from the input directory"""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        return image_files
    
    def augment_dataset(self):
        """Main function to augment the entire dataset"""
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No image files found in {self.input_dir}")
            return
        
        print(f"Found {len(image_files)} images in the dataset")
        print(f"Target augmentation factor: {self.augmentation_factor}x")
        print(f"Expected output: {len(image_files) * self.augmentation_factor} images")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        # Copy original images first
        print("Copying original images...")
        for img_file in tqdm(image_files, desc="Copying originals"):
            shutil.copy2(img_file, self.output_dir / f"original_{img_file.name}")
        
        # Generate augmented images
        augmentation_count = 0
        
        for img_file in tqdm(image_files, desc="Augmenting images"):
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"Warning: Could not load {img_file}")
                    continue
                
                base_name = img_file.stem
                ext = img_file.suffix.lower()
                
                # Generate augmented versions
                augmented_images = self.generate_augmentations(image)
                
                # Save augmented images
                for i, aug_img in enumerate(augmented_images):
                    if i >= self.augmentation_factor - 1:  # -1 because we already copied original
                        break
                    
                    output_name = f"{base_name}_aug_{i+1}{ext}"
                    output_path = self.output_dir / output_name
                    
                    cv2.imwrite(str(output_path), aug_img)
                    augmentation_count += 1
                    
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
        
        print(f"\nAugmentation complete!")
        print(f"Original images: {len(image_files)}")
        print(f"Augmented images generated: {augmentation_count}")
        print(f"Total images in output: {len(image_files) + augmentation_count}")
    
    def generate_augmentations(self, image):
        """Generate multiple augmented versions of a single image"""
        augmented_images = []
        
        # Define augmentation techniques
        techniques = [
            self.horizontal_flip,
            self.vertical_flip,
            self.rotate_image,
            self.random_crop,
            self.brightness_adjustment,
            self.contrast_adjustment,
            self.saturation_adjustment,
            self.gaussian_blur,
            self.sharpen_filter,
            self.noise_addition,
            self.color_shift,
            self.zoom_in,
            self.perspective_transform,
            self.elastic_transform
        ]
        
        # Generate required number of augmentations
        for i in range(self.augmentation_factor - 1):  # -1 for original
            # Select random techniques (can combine multiple)
            selected_techniques = random.sample(techniques, random.randint(1, 3))
            
            # Apply techniques sequentially
            aug_image = image.copy()
            for technique in selected_techniques:
                aug_image = technique(aug_image)
            
            augmented_images.append(aug_image)
        
        return augmented_images
    
    # MIRRORING TECHNIQUES
    def horizontal_flip(self, image):
        """Horizontal mirroring"""
        return cv2.flip(image, 1)
    
    def vertical_flip(self, image):
        """Vertical mirroring"""
        return cv2.flip(image, 0)
    
    # ROTATION TECHNIQUES
    def rotate_image(self, image):
        """Random rotation between -30 to 30 degrees"""
        angle = random.uniform(-30, 30)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    
    # CROPPING TECHNIQUES
    def random_crop(self, image):
        """Random cropping with resizing back to original dimensions"""
        height, width = image.shape[:2]
        
        # Random crop parameters (keep 70-95% of original size)
        crop_ratio = random.uniform(0.7, 0.95)
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)
        
        # Random starting position
        start_y = random.randint(0, height - crop_height)
        start_x = random.randint(0, width - crop_width)
        
        # Crop and resize back
        cropped = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
        resized = cv2.resize(cropped, (width, height))
        
        return resized
    
    def zoom_in(self, image):
        """Zoom in effect"""
        height, width = image.shape[:2]
        
        # Zoom parameters
        zoom_factor = random.uniform(1.1, 1.3)
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        
        # Resize and center crop
        zoomed = cv2.resize(image, (new_width, new_height))
        
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        
        return zoomed[start_y:start_y+height, start_x:start_x+width]
    
    # COLOR AUGMENTATION TECHNIQUES
    def brightness_adjustment(self, image):
        """Adjust brightness"""
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Random brightness factor
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def contrast_adjustment(self, image):
        """Adjust contrast"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def saturation_adjustment(self, image):
        """Adjust color saturation"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(factor)
        
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def color_shift(self, image):
        """Shift colors in HSV space"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Random hue shift
        hue_shift = random.randint(-10, 10)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Convert back
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # FILTER TECHNIQUES
    def gaussian_blur(self, image):
        """Apply Gaussian blur"""
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def sharpen_filter(self, image):
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def noise_addition(self, image):
        """Add random noise"""
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    # ADVANCED GEOMETRIC TRANSFORMATIONS
    def perspective_transform(self, image):
        """Apply random perspective transformation"""
        height, width = image.shape[:2]
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # Add random distortion to destination points
        distortion = 20
        dst_points = np.float32([
            [random.randint(0, distortion), random.randint(0, distortion)],
            [width - random.randint(0, distortion), random.randint(0, distortion)],
            [width - random.randint(0, distortion), height - random.randint(0, distortion)],
            [random.randint(0, distortion), height - random.randint(0, distortion)]
        ])
        
        # Get perspective transform matrix and apply
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (width, height))
    
    def elastic_transform(self, image):
        """Apply elastic transformation"""
        height, width = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (height//10, width//10)) * 10
        dy = np.random.uniform(-1, 1, (height//10, width//10)) * 10
        
        # Resize displacement fields to image size
        dx = cv2.resize(dx, (width, height))
        dy = cv2.resize(dy, (width, height))
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap the image
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Image Dataset Augmentation Tool')
    parser.add_argument('--input_dir', type=str, 
                       default=r'D:\Downloads\Code_Autonomous\Project_AIMS\crowd_wala_dataset\train_data\images',
                       help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, 
                       default=r'D:\Downloads\Code_Autonomous\Project_AIMS\crowd_wala_dataset\train_data\augmented_images',
                       help='Output directory for augmented images')
    parser.add_argument('--augmentation_factor', type=int, default=5,
                       help='How many times to expand the dataset (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize and run augmentation
    print("=== Image Dataset Augmentation Tool ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentation factor: {args.augmentation_factor}x")
    print()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    # Create augmentator and run
    augmentator = ImageAugmentator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augmentation_factor=args.augmentation_factor
    )
    
    augmentator.augment_dataset()
    
    print("\n=== Augmentation Summary ===")
    print("Applied techniques include:")
    print("✓ Mirroring (horizontal & vertical flips)")
    print("✓ Rotation (random angles)")
    print("✓ Cropping (random crops with resize)")
    print("✓ Color augmentation (brightness, contrast, saturation)")
    print("✓ Filters (blur, sharpen, noise)")
    print("✓ Advanced transformations (perspective, elastic)")
    print(f"\nCheck your augmented dataset at: {args.output_dir}")


# Utility function for batch processing
def quick_augment(input_dir, output_dir=None, factor=5):
    """Quick augmentation function for Jupyter notebooks or direct usage"""
    if output_dir is None:
        output_dir = str(Path(input_dir).parent / "augmented_images")
    
    augmentator = ImageAugmentator(input_dir, output_dir, factor)
    augmentator.augment_dataset()
    return output_dir


if __name__ == "__main__":
    # If running directly with your specific path
    input_path = r"D:\Downloads\Code_Autonomous\Project_AIMS\crowd_wala_dataset\train_data\images"
    output_path = r"D:\Downloads\Code_Autonomous\Project_AIMS\crowd_wala_dataset\train_data\augmented_images"
    
    print("=== Quick Start - Image Augmentation ===")
    print(f"Processing images from: {input_path}")
    print(f"Saving augmented images to: {output_path}")
    print()
    
    # Run augmentation
    augmentator = ImageAugmentator(
        input_dir=input_path,
        output_dir=output_path,
        augmentation_factor=5  # 5x expansion
    )
    
    augmentator.augment_dataset()