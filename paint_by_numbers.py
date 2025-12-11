"""
Paint by Numbers Image Processor
Converts any image into a paint-by-numbers template with N colors
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import json

class PaintByNumbers:
    def __init__(self, image_path, n_colors=20, min_region_size=10):
        """
        Initialize the Paint by Numbers converter
        
        Args:
            image_path: Path to input image
            n_colors: Number of colors to reduce the image to
            min_region_size: Minimum pixel count for a region (smaller regions get merged)
        """
        self.image_path = image_path
        self.n_colors = n_colors
        self.min_region_size = min_region_size
        
        # Load image
        self.original_image = cv2.imread(image_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Store results
        self.palette = None
        self.quantized_image = None
        self.labeled_regions = None
        self.template = None
        self.color_counts = None
        
    def quantize_colors(self):
        """
        Reduce image to N colors using K-means clustering in LAB color space
        LAB is perceptually uniform, so similar colors cluster better
        """
        print(f"Quantizing image to {self.n_colors} colors...")
        
        # Convert to LAB color space (more perceptually uniform than RGB)
        lab_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LAB)
        
        # Reshape image to be a list of pixels
        pixels = lab_image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (these are our N colors in LAB space)
        centers_lab = kmeans.cluster_centers_
        
        # Convert centers back to RGB for display
        centers_lab_reshaped = centers_lab.reshape(1, -1, 3).astype(np.uint8)
        centers_rgb = cv2.cvtColor(centers_lab_reshaped, cv2.COLOR_LAB2RGB)
        self.palette = centers_rgb.reshape(-1, 3)
        
        # Replace each pixel with its cluster center
        labels = kmeans.labels_
        quantized_lab = centers_lab[labels]
        quantized_lab = quantized_lab.reshape(lab_image.shape).astype(np.uint8)
        
        # Convert back to RGB
        self.quantized_image = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2RGB)
        
        # Store labels reshaped to image dimensions
        self.label_map = labels.reshape(self.original_image.shape[:2])
        
        # Count pixels per color
        self.color_counts = Counter(labels)
        
        # Sort palette by frequency (most common colors first)
        sorted_indices = sorted(range(self.n_colors), 
                               key=lambda i: self.color_counts[i], 
                               reverse=True)
        
        # Reorder palette
        old_palette = self.palette.copy()
        old_label_map = self.label_map.copy()
        
        for new_idx, old_idx in enumerate(sorted_indices):
            self.palette[new_idx] = old_palette[old_idx]
            self.label_map[old_label_map == old_idx] = new_idx + 1000  # Temporary offset
        
        self.label_map -= 1000  # Remove offset
        
        return self.palette, self.quantized_image
    
    def create_regions(self):
        """
        Create distinct regions for each color using connected component analysis
        This segments the image into paintable regions
        """
        print("Creating numbered regions...")
        
        h, w = self.label_map.shape
        self.labeled_regions = np.zeros((h, w), dtype=np.int32)
        
        current_label = 1
        
        # For each color, find connected components
        for color_idx in range(self.n_colors):
            # Create binary mask for this color
            mask = (self.label_map == color_idx).astype(np.uint8)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(mask)
            
            # Assign unique labels to each component
            for component_idx in range(1, num_labels):  # Skip 0 (background)
                component_mask = (labels == component_idx)
                component_size = np.sum(component_mask)
                
                # Only keep regions larger than minimum size
                if component_size >= self.min_region_size:
                    self.labeled_regions[component_mask] = current_label
                    current_label += 1
        
        # Merge small regions into neighboring regions
        self._merge_small_regions()
        
        return self.labeled_regions
    
    def _merge_small_regions(self):
        """
        Merge small regions into their largest neighbor
        """
        unique_labels = np.unique(self.labeled_regions)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        for label in unique_labels:
            mask = (self.labeled_regions == label)
            size = np.sum(mask)
            
            if size < self.min_region_size:
                # Find neighboring labels
                dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3,3), np.uint8))
                border = dilated & ~mask
                neighbors = self.labeled_regions[border]
                neighbors = neighbors[neighbors > 0]
                
                if len(neighbors) > 0:
                    # Merge into most common neighbor
                    most_common_neighbor = Counter(neighbors).most_common(1)[0][0]
                    self.labeled_regions[mask] = most_common_neighbor
    
    def create_template(self, font_size=8, show_numbers=True):
        """
        Create the paint-by-numbers template with numbers and outlines
        
        Args:
            font_size: Size of numbers on the template
            show_numbers: Whether to show numbers in each region
        """
        print("Creating paint-by-numbers template...")
        
        h, w = self.labeled_regions.shape
        
        # Create white background
        template = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw region boundaries
        boundaries = cv2.Canny(self.labeled_regions.astype(np.uint8), 0, 1)
        boundaries = cv2.dilate(boundaries, np.ones((2,2), np.uint8))
        template[boundaries > 0] = [0, 0, 0]  # Black lines
        
        if show_numbers:
            # Convert to PIL for text rendering
            pil_template = Image.fromarray(template)
            draw = ImageDraw.Draw(pil_template)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # For each region, place a number at its center
            unique_labels = np.unique(self.labeled_regions)
            unique_labels = unique_labels[unique_labels > 0]
            
            for label in unique_labels:
                mask = (self.labeled_regions == label)
                
                # Find region centroid
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0:
                    continue
                    
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                
                # Get color index for this region
                color_idx = self.label_map[center_y, center_x]
                
                # Draw number (1-indexed for user friendliness)
                text = str(color_idx + 1)
                
                # Get text size for centering
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2
                
                # Draw white background for better visibility
                padding = 2
                draw.rectangle([text_x - padding, text_y - padding,
                               text_x + text_width + padding, text_y + text_height + padding],
                              fill='white')
                
                draw.text((text_x, text_y), text, fill='black', font=font)
            
            template = np.array(pil_template)
        
        self.template = template
        return template
    
    def match_to_paint_colors(self, paint_database=None):
        """
        Match extracted colors to commercial paint colors
        
        Args:
            paint_database: Dictionary of paint colors {name: [R, G, B]}
                          If None, uses a basic set of common colors
        """
        if paint_database is None:
            # Basic paint color database (you can expand this)
            paint_database = {
                "Titanium White": [255, 255, 255],
                "Ivory Black": [41, 36, 33],
                "Cadmium Yellow": [255, 236, 0],
                "Yellow Ochre": [227, 168, 87],
                "Cadmium Red": [227, 0, 34],
                "Alizarin Crimson": [227, 38, 54],
                "Burnt Sienna": [138, 54, 15],
                "Burnt Umber": [138, 51, 36],
                "Ultramarine Blue": [18, 10, 143],
                "Prussian Blue": [0, 49, 83],
                "Cerulean Blue": [42, 82, 190],
                "Viridian Green": [64, 130, 109],
                "Sap Green": [48, 98, 48],
                "Chrome Green": [56, 97, 39],
                "Raw Umber": [115, 74, 18],
                "Raw Sienna": [199, 97, 20],
                "Payne's Gray": [83, 104, 120],
                "Vermilion": [227, 66, 52],
                "Magenta": [202, 31, 123],
                "Violet": [143, 0, 255],
            }
        
        print("Matching colors to paint database...")
        
        matched_colors = []
        
        for i, color_rgb in enumerate(self.palette):
            best_match = None
            best_distance = float('inf')
            
            # Find closest color in database using Euclidean distance
            for paint_name, paint_rgb in paint_database.items():
                dist = distance.euclidean(color_rgb, paint_rgb)
                
                if dist < best_distance:
                    best_distance = dist
                    best_match = {
                        'number': i + 1,
                        'extracted_rgb': color_rgb.tolist(),
                        'extracted_hex': self._rgb_to_hex(color_rgb),
                        'paint_name': paint_name,
                        'paint_rgb': paint_rgb,
                        'paint_hex': self._rgb_to_hex(paint_rgb),
                        'distance': round(dist, 2),
                        'pixel_count': self.color_counts[i]
                    }
            
            matched_colors.append(best_match)
        
        return matched_colors
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def visualize_results(self, save_path=None):
        """
        Create a visualization showing all results
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Original image
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Quantized image (preview)
        axes[0, 1].imshow(self.quantized_image)
        axes[0, 1].set_title(f'Quantized to {self.n_colors} Colors', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Paint-by-numbers template
        axes[1, 0].imshow(self.template)
        axes[1, 0].set_title('Paint-by-Numbers Template', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Color palette
        palette_img = np.ones((100, self.n_colors * 50, 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            palette_img[:, i*50:(i+1)*50] = color
        
        axes[1, 1].imshow(palette_img)
        axes[1, 1].set_title('Color Palette', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Add color numbers below palette
        for i in range(self.n_colors):
            axes[1, 1].text(i*50 + 25, 110, str(i+1), 
                          ha='center', va='top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def save_color_guide(self, matched_colors, save_path):
        """
        Save a detailed color guide as JSON
        """
        guide = {
            'total_colors': self.n_colors,
            'colors': matched_colors,
            'instructions': 'Match the numbers on the canvas to the paint colors below'
        }
        
        with open(save_path, 'w') as f:
            json.dump(guide, f, indent=2)
        
        print(f"Color guide saved to {save_path}")
        
    def create_color_reference_sheet(self, matched_colors, save_path):
        """
        Create a visual color reference sheet showing extracted colors and paint matches
        """
        fig, ax = plt.subplots(figsize=(12, max(8, self.n_colors * 0.5)))
        
        ax.axis('off')
        
        # Title
        fig.suptitle('Paint-by-Numbers Color Guide', fontsize=16, fontweight='bold', y=0.98)
        
        # Create color swatches
        for i, color_info in enumerate(matched_colors):
            y_pos = 0.9 - (i * 0.85 / self.n_colors)
            
            # Extracted color swatch
            rect1 = plt.Rectangle((0.05, y_pos), 0.08, 0.7/self.n_colors, 
                                 facecolor=np.array(color_info['extracted_rgb'])/255)
            ax.add_patch(rect1)
            
            # Paint match swatch
            rect2 = plt.Rectangle((0.15, y_pos), 0.08, 0.7/self.n_colors,
                                 facecolor=np.array(color_info['paint_rgb'])/255)
            ax.add_patch(rect2)
            
            # Text information
            text = f"#{color_info['number']}: {color_info['paint_name']}\n"
            text += f"Match: {color_info['extracted_hex']} → {color_info['paint_hex']}\n"
            text += f"Pixels: {color_info['pixel_count']:,}"
            
            ax.text(0.25, y_pos + 0.35/self.n_colors, text,
                   fontsize=9, va='center')
        
        # Legend
        ax.text(0.05, 0.95, 'Extracted', fontsize=10, fontweight='bold', ha='left')
        ax.text(0.15, 0.95, 'Paint Match', fontsize=10, fontweight='bold', ha='left')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Color reference sheet saved to {save_path}")
        plt.show()
        
        return fig
    
    def save_template(self, save_path):
        """Save the template image"""
        template_rgb = cv2.cvtColor(self.template, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, template_rgb)
        print(f"Template saved to {save_path}")
    
    def process_all(self, output_dir='/home/claude/output'):
        """
        Run the complete paint-by-numbers conversion pipeline
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Quantize colors
        self.quantize_colors()
        
        # Step 2: Create regions
        self.create_regions()
        
        # Step 3: Create template
        self.create_template()
        
        # Step 4: Match to paint colors
        matched_colors = self.match_to_paint_colors()
        
        # Step 5: Save all outputs
        self.visualize_results(f'{output_dir}/visualization.png')
        self.save_template(f'{output_dir}/template.png')
        self.save_color_guide(matched_colors, f'{output_dir}/color_guide.json')
        self.create_color_reference_sheet(matched_colors, f'{output_dir}/color_reference.png')
        
        print(f"\n✓ All files saved to {output_dir}/")
        print(f"  - visualization.png: Overview of all results")
        print(f"  - template.png: Paint-by-numbers template")
        print(f"  - color_guide.json: Detailed color matching data")
        print(f"  - color_reference.png: Visual color guide")
        
        return matched_colors


if __name__ == "__main__":
    print("Paint by Numbers Converter")
    print("=" * 80)
    print("\nUsage:")
    print("  pbn = PaintByNumbers('your_image.jpg', n_colors=20)")
    print("  pbn.process_all()")
    print("\n" + "=" * 80)
