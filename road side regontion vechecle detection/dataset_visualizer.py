"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     DATASET VISUALIZATION & ANALYSIS TOOL                      ║
║                  Traffic Sign Recognition Dataset Explorer                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script provides comprehensive visualization and analysis of your traffic
sign recognition dataset. It generates:
- Class distribution charts
- Sample images from each class
- Dataset statistics
- Training history graphs (if available)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import Counter
import seaborn as sns

# Set style for beautiful plots
plt.style.use('dark_background')
sns.set_palette("husl")

# ============================================================================
# TRAFFIC SIGN CLASSES (German Traffic Sign Recognition Dataset - 43 classes)
# ============================================================================
TRAFFIC_SIGN_CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing (3.5t+)",
    11: "Right-of-way",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles 3.5t+ prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals",
    32: "End all limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight/right",
    37: "Go straight/left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout",
    41: "End no passing",
    42: "End no passing (3.5t+)"
}


def analyze_dataset(dataset_path):
    """
    Analyze the dataset and return statistics.
    
    Args:
        dataset_path: Path to the dataset folder (should contain subfolders for each class)
    
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        "total_images": 0,
        "classes": {},
        "class_counts": [],
        "image_sizes": [],
        "class_names": []
    }
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return None
    
    # Iterate through class folders
    for class_folder in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_folder)
        
        if os.path.isdir(class_path):
            # Count images in this class
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.ppm'))]
            
            count = len(images)
            stats["total_images"] += count
            
            try:
                class_id = int(class_folder)
                class_name = TRAFFIC_SIGN_CLASSES.get(class_id, f"Class {class_id}")
            except ValueError:
                class_id = class_folder
                class_name = class_folder
            
            stats["classes"][class_id] = {
                "name": class_name,
                "count": count,
                "path": class_path,
                "sample_images": images[:5] if images else []
            }
            
            stats["class_counts"].append(count)
            stats["class_names"].append(class_name[:20] + "..." if len(class_name) > 20 else class_name)
            
            # Get sample image sizes
            if images:
                try:
                    sample_img = Image.open(os.path.join(class_path, images[0]))
                    stats["image_sizes"].append(sample_img.size)
                except:
                    pass
    
    return stats


def plot_class_distribution(stats, save_path=None):
    """
    Create a beautiful bar chart showing class distribution.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#1a1a2e')
    
    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stats["class_counts"])))
    
    # Bar plot
    bars = ax.barh(range(len(stats["class_counts"])), stats["class_counts"], 
                  color=colors, edgecolor='white', linewidth=0.5)
    
    # Customize
    ax.set_yticks(range(len(stats["class_names"])))
    ax.set_yticklabels(stats["class_names"], fontsize=9, color='white')
    ax.set_xlabel("Number of Images", fontsize=12, color='white')
    ax.set_title("📊 Traffic Sign Dataset - Class Distribution", 
                fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, stats["class_counts"])):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
               str(count), va='center', fontsize=8, color='white')
    
    # Grid
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_pie_chart(stats, save_path=None):
    """
    Create a pie chart showing top 10 classes.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    
    # Get top 10 classes
    sorted_indices = np.argsort(stats["class_counts"])[-10:]
    top_counts = [stats["class_counts"][i] for i in sorted_indices]
    top_names = [stats["class_names"][i] for i in sorted_indices]
    
    # Add "Others" category
    other_count = sum(stats["class_counts"]) - sum(top_counts)
    if other_count > 0:
        top_counts.append(other_count)
        top_names.append("Others")
    
    # Colors
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(top_counts)))
    
    # Explode the largest slice
    explode = [0.05 if i == len(top_counts)-2 else 0 for i in range(len(top_counts))]
    
    wedges, texts, autotexts = ax.pie(top_counts, labels=top_names, autopct='%1.1f%%',
                                       colors=colors, explode=explode,
                                       textprops={'color': 'white', 'fontsize': 9},
                                       pctdistance=0.8)
    
    ax.set_title("🥧 Top 10 Classes by Sample Count", 
                fontsize=16, fontweight='bold', color='white', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_sample_images(stats, save_path=None):
    """
    Display sample images from each class.
    """
    num_classes = min(12, len(stats["classes"]))
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f23')
    fig.suptitle("🖼️ Sample Traffic Signs from Dataset", 
                fontsize=16, fontweight='bold', color='white')
    
    class_items = list(stats["classes"].items())[:num_classes]
    
    for idx, (class_id, class_info) in enumerate(class_items):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        ax.set_facecolor('#1a1a2e')
        
        if class_info["sample_images"]:
            try:
                img_path = os.path.join(class_info["path"], class_info["sample_images"][0])
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(class_info["name"][:25], fontsize=9, color='white')
            except:
                ax.text(0.5, 0.5, "Error\nloading", ha='center', va='center', 
                       color='gray', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No\nimages", ha='center', va='center', 
                   color='gray', fontsize=10)
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_classes, 12):
        row, col = idx // 4, idx % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_statistics_summary(stats, save_path=None):
    """
    Create a comprehensive statistics summary visualization.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f23')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    
    # 1. Total Statistics Box
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    total_images = stats["total_images"]
    num_classes = len(stats["classes"])
    avg_per_class = total_images // num_classes if num_classes > 0 else 0
    
    text_content = f"""
    📊 DATASET OVERVIEW
    ─────────────────
    Total Images: {total_images:,}
    Number of Classes: {num_classes}
    Avg Images/Class: {avg_per_class}
    """
    
    ax1.text(0.1, 0.5, text_content, transform=ax1.transAxes,
            fontsize=14, color='white', verticalalignment='center',
            fontfamily='monospace')
    ax1.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, 
                                edgecolor='#667eea', linewidth=2))
    
    # 2. Histogram of class sizes
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor('#1a1a2e')
    ax2.hist(stats["class_counts"], bins=20, color='#667eea', edgecolor='white', alpha=0.8)
    ax2.set_xlabel("Images per Class", color='white')
    ax2.set_ylabel("Number of Classes", color='white')
    ax2.set_title("📈 Distribution of Class Sizes", color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.xaxis.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Top 10 largest classes (horizontal bar)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor('#1a1a2e')
    
    sorted_indices = np.argsort(stats["class_counts"])[-10:]
    top_counts = [stats["class_counts"][i] for i in sorted_indices]
    top_names = [stats["class_names"][i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
    bars = ax3.barh(range(10), top_counts, color=colors)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(top_names, fontsize=9, color='white')
    ax3.set_xlabel("Number of Images", color='white')
    ax3.set_title("🏆 Top 10 Classes by Size", color='white', fontweight='bold')
    ax3.tick_params(colors='white')
    
    for bar, count in zip(bars, top_counts):
        ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=9, color='white')
    
    # 4. Min/Max comparison
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('#1a1a2e')
    
    min_count = min(stats["class_counts"])
    max_count = max(stats["class_counts"])
    min_idx = stats["class_counts"].index(min_count)
    max_idx = stats["class_counts"].index(max_count)
    
    categories = ['Smallest\nClass', 'Largest\nClass']
    values = [min_count, max_count]
    colors = ['#ff6b6b', '#00d9a5']
    
    bars = ax4.bar(categories, values, color=colors, width=0.5)
    ax4.set_ylabel("Number of Images", color='white')
    ax4.set_title("⚖️ Class Size Range", color='white', fontweight='bold')
    ax4.tick_params(colors='white')
    
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', fontsize=12, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.show()


def main():
    """
    Main function to run dataset visualization.
    """
    print("=" * 60)
    print("   TRAFFIC SIGN DATASET VISUALIZATION TOOL")
    print("=" * 60)
    
    # Ask for dataset path
    print("\nPlease enter the path to your training dataset folder.")
    print("(The folder should contain subfolders for each class, numbered 0-42)")
    print("\nExample: D:\\datasets\\traffic_signs\\train")
    
    dataset_path = input("\nDataset Path: ").strip()
    
    if not dataset_path:
        # Try common paths
        common_paths = [
            r"d:\road side regontion vechecle detection\train",
            r"d:\road side regontion vechecle detection\Train",
            r"d:\road side regontion vechecle detection\dataset",
            r"d:\road side regontion vechecle detection\data\train"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"Found dataset at: {dataset_path}")
                break
    
    if not os.path.exists(dataset_path):
        print(f"\n[ERROR] Dataset path not found: {dataset_path}")
        print("\nPlease ensure you have the dataset in the correct location.")
        return
    
    print(f"\n[OK] Analyzing dataset: {dataset_path}")
    
    # Analyze dataset
    stats = analyze_dataset(dataset_path)
    
    if not stats or not stats["classes"]:
        print("[ERROR] No data found in the dataset folder.")
        return
    
    # Print summary
    print("\n" + "=" * 40)
    print("   DATASET SUMMARY")
    print("=" * 40)
    print(f"Total Images: {stats['total_images']:,}")
    print(f"Number of Classes: {len(stats['classes'])}")
    print(f"Average Images per Class: {stats['total_images'] // len(stats['classes'])}")
    print(f"Min Images in a Class: {min(stats['class_counts'])}")
    print(f"Max Images in a Class: {max(stats['class_counts'])}")
    
    # Save path
    save_dir = os.path.dirname(dataset_path) or "."
    
    # Generate visualizations
    print("\n[INFO] Generating visualizations...")
    
    try:
        plot_class_distribution(stats, os.path.join(save_dir, "class_distribution.png"))
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
    
    try:
        plot_pie_chart(stats, os.path.join(save_dir, "class_pie_chart.png"))
    except Exception as e:
        print(f"Error creating pie chart: {e}")
    
    try:
        plot_sample_images(stats, os.path.join(save_dir, "sample_images.png"))
    except Exception as e:
        print(f"Error creating sample images: {e}")
    
    try:
        plot_statistics_summary(stats, os.path.join(save_dir, "statistics_summary.png"))
    except Exception as e:
        print(f"Error creating statistics summary: {e}")
    
    print("\n[DONE] Visualization complete!")
    print(f"Charts saved to: {save_dir}")


if __name__ == "__main__":
    main()
