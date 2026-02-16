"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SAMPLE DATASET VISUALIZATION                               ║
║                  Traffic Sign Recognition - Demo Charts                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script creates sample visualizations for the 43 traffic sign classes
based on typical GTSRB dataset distributions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set style for beautiful plots
plt.style.use('dark_background')

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
    6: "End speed limit (80km/h)",
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
    30: "Beware ice/snow",
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

# Typical GTSRB dataset distribution (approximate sample counts)
SAMPLE_DISTRIBUTION = [
    210, 2220, 2250, 1410, 1980, 1860, 420, 1440, 1410, 1470,
    2010, 1320, 2100, 2160, 780, 630, 420, 1110, 1200, 210,
    360, 330, 390, 510, 270, 1500, 600, 240, 540, 270,
    450, 780, 240, 689, 420, 1200, 390, 210, 2070, 300,
    360, 240, 240
]

def create_model_architecture_diagram(save_path=None):
    """
    Create a visual diagram of the CNN model architecture.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # Title
    ax.text(50, 47, '🧠 CNN Model Architecture', fontsize=20, 
            fontweight='bold', ha='center', color='white')
    ax.text(50, 44, 'Traffic Sign Recognition Neural Network', fontsize=12, 
            ha='center', color='#a0a0b0')
    
    # Layer boxes
    layers = [
        {'name': 'Input\n30x30x3', 'x': 5, 'color': '#667eea', 'width': 8},
        {'name': 'Conv2D\n32 filters\n5x5', 'x': 16, 'color': '#764ba2', 'width': 8},
        {'name': 'Conv2D\n32 filters\n5x5', 'x': 27, 'color': '#764ba2', 'width': 8},
        {'name': 'MaxPool\n2x2', 'x': 38, 'color': '#00d9a5', 'width': 7},
        {'name': 'Conv2D\n64 filters\n3x3', 'x': 48, 'color': '#764ba2', 'width': 8},
        {'name': 'Conv2D\n64 filters\n3x3', 'x': 59, 'color': '#764ba2', 'width': 8},
        {'name': 'MaxPool\n2x2', 'x': 70, 'color': '#00d9a5', 'width': 7},
        {'name': 'Dense\n256', 'x': 80, 'color': '#ffc107', 'width': 7},
        {'name': 'Output\n43 classes', 'x': 90, 'color': '#ff6b6b', 'width': 8}
    ]
    
    for layer in layers:
        rect = mpatches.FancyBboxPatch((layer['x'], 15), layer['width'], 20,
                                       boxstyle="round,pad=0.02,rounding_size=1",
                                       facecolor=layer['color'], edgecolor='white',
                                       linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(layer['x'] + layer['width']/2, 25, layer['name'],
               ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Arrows
    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + layers[i]['width']
        x2 = layers[i+1]['x']
        ax.annotate('', xy=(x2, 25), xytext=(x1, 25),
                   arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    
    # Dropout annotations
    ax.text(32, 10, 'Dropout 25%', fontsize=8, ha='center', color='#a0a0b0')
    ax.text(64, 10, 'Dropout 25%', fontsize=8, ha='center', color='#a0a0b0')
    ax.text(83.5, 10, 'Dropout 50%', fontsize=8, ha='center', color='#a0a0b0')
    
    # Legend
    legend_items = [
        ('Input Layer', '#667eea'),
        ('Convolution', '#764ba2'),
        ('Pooling', '#00d9a5'),
        ('Dense Layer', '#ffc107'),
        ('Output', '#ff6b6b')
    ]
    
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(mpatches.Rectangle((5 + i*18, 3), 3, 2, facecolor=color, edgecolor='white'))
        ax.text(9 + i*18, 4, label, fontsize=9, va='center', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_class_distribution_chart(save_path=None):
    """
    Create a horizontal bar chart showing class distribution.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#1a1a2e')
    
    # Prepare data
    class_names = [TRAFFIC_SIGN_CLASSES[i][:25] for i in range(43)]
    counts = SAMPLE_DISTRIBUTION
    
    # Create gradient colors based on count
    norm_counts = np.array(counts) / max(counts)
    colors = plt.cm.viridis(0.2 + 0.7 * norm_counts)
    
    # Bar plot
    y_pos = np.arange(len(class_names))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='white', linewidth=0.3, height=0.8)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8, color='white')
    ax.set_xlabel("Number of Training Images", fontsize=12, color='white', labelpad=10)
    ax.set_title("📊 Traffic Sign Dataset - Class Distribution\n(Based on GTSRB Dataset)", 
                fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
               f'{count:,}', va='center', fontsize=7, color='white')
    
    # Grid
    ax.xaxis.grid(True, alpha=0.3, linestyle='--', color='white')
    ax.set_axisbelow(True)
    ax.tick_params(colors='white')
    
    # Invert y-axis to show class 0 at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_category_pie_chart(save_path=None):
    """
    Create a pie chart showing sign categories.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    
    # Group signs by category
    categories = {
        'Speed Limits': sum(SAMPLE_DISTRIBUTION[0:9]),
        'Prohibitory': sum(SAMPLE_DISTRIBUTION[9:18]),
        'Warning Signs': sum(SAMPLE_DISTRIBUTION[18:32]),
        'Mandatory': sum(SAMPLE_DISTRIBUTION[32:43])
    }
    
    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = ['#667eea', '#ff6b6b', '#ffc107', '#00d9a5']
    explode = (0.02, 0.02, 0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, explode=explode,
                                       textprops={'color': 'white', 'fontsize': 12},
                                       pctdistance=0.75,
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title("🥧 Traffic Sign Categories Distribution", 
                fontsize=18, fontweight='bold', color='white', pad=20)
    
    # Add legend with counts
    legend_labels = [f'{label} ({count:,} images)' for label, count in categories.items()]
    ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1),
             ncol=2, fontsize=10, facecolor='#1a1a2e', edgecolor='white',
             labelcolor='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_statistics_dashboard(save_path=None):
    """
    Create a comprehensive statistics dashboard.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f23')
    fig.suptitle('📈 Traffic Sign Recognition - Dataset Statistics Dashboard', 
                fontsize=18, fontweight='bold', color='white', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Key Metrics (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    ax1.axis('off')
    
    total = sum(SAMPLE_DISTRIBUTION)
    metrics_text = f"""
    ╔═══════════════════════╗
    ║   DATASET OVERVIEW    ║
    ╠═══════════════════════╣
    ║ Total Images: {total:,}  
    ║ Classes: 43           
    ║ Avg per Class: {total//43:,}  
    ║ Min Images: {min(SAMPLE_DISTRIBUTION):,}     
    ║ Max Images: {max(SAMPLE_DISTRIBUTION):,}   
    ╚═══════════════════════╝
    """
    ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes,
            fontsize=12, color='#00d9a5', verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace')
    
    # 2. Model Info (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')
    ax2.axis('off')
    
    model_text = f"""
    ╔═══════════════════════╗
    ║    MODEL DETAILS      ║
    ╠═══════════════════════╣
    ║ Input Size: 30x30x3   
    ║ Architecture: CNN     
    ║ Conv Layers: 4        
    ║ Dense Layers: 2       
    ║ Parameters: 242,253   
    ║ Output: 43 classes    
    ╚═══════════════════════╝
    """
    ax2.text(0.5, 0.5, model_text, transform=ax2.transAxes,
            fontsize=12, color='#667eea', verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace')
    
    # 3. Top 5 Classes (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#1a1a2e')
    
    top5_idx = np.argsort(SAMPLE_DISTRIBUTION)[-5:][::-1]
    top5_names = [TRAFFIC_SIGN_CLASSES[i][:20] for i in top5_idx]
    top5_counts = [SAMPLE_DISTRIBUTION[i] for i in top5_idx]
    
    colors = ['#ffc107', '#ffca28', '#ffd54f', '#ffe082', '#ffecb3']
    bars = ax3.barh(range(5), top5_counts, color=colors, edgecolor='white')
    ax3.set_yticks(range(5))
    ax3.set_yticklabels(top5_names, fontsize=9, color='white')
    ax3.set_title('🏆 Top 5 Classes', color='white', fontsize=12, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.invert_yaxis()
    
    for bar, count in zip(bars, top5_counts):
        ax3.text(bar.get_width() - 100, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10, color='black', fontweight='bold')
    
    # 4. Histogram (Bottom Left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#1a1a2e')
    ax4.hist(SAMPLE_DISTRIBUTION, bins=15, color='#764ba2', edgecolor='white', alpha=0.8)
    ax4.set_xlabel('Images per Class', color='white', fontsize=10)
    ax4.set_ylabel('Number of Classes', color='white', fontsize=10)
    ax4.set_title('📊 Class Size Distribution', color='white', fontsize=12, fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.xaxis.grid(True, alpha=0.3, linestyle='--')
    
    # 5. Category Breakdown (Bottom Center + Right)
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_facecolor('#1a1a2e')
    
    categories = ['Speed Limits\n(0-8)', 'Prohibitory\n(9-17)', 
                 'Warning\n(18-31)', 'Mandatory\n(32-42)']
    cat_counts = [
        sum(SAMPLE_DISTRIBUTION[0:9]),
        sum(SAMPLE_DISTRIBUTION[9:18]),
        sum(SAMPLE_DISTRIBUTION[18:32]),
        sum(SAMPLE_DISTRIBUTION[32:43])
    ]
    colors = ['#667eea', '#ff6b6b', '#ffc107', '#00d9a5']
    
    bars = ax5.bar(categories, cat_counts, color=colors, edgecolor='white', width=0.6)
    ax5.set_ylabel('Total Images', color='white', fontsize=10)
    ax5.set_title('📦 Images by Sign Category', color='white', fontsize=12, fontweight='bold')
    ax5.tick_params(colors='white')
    
    for bar, count in zip(bars, cat_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', fontsize=11, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
        print(f"Saved: {save_path}")
    
    plt.show()


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("   GENERATING TRAFFIC SIGN DATASET VISUALIZATIONS")
    print("=" * 60)
    
    save_dir = r"d:\road side regontion vechecle detection"
    
    print("\n[1/4] Creating Model Architecture Diagram...")
    create_model_architecture_diagram(os.path.join(save_dir, "model_architecture.png"))
    
    print("\n[2/4] Creating Class Distribution Chart...")
    create_class_distribution_chart(os.path.join(save_dir, "class_distribution.png"))
    
    print("\n[3/4] Creating Category Pie Chart...")
    create_category_pie_chart(os.path.join(save_dir, "category_pie_chart.png"))
    
    print("\n[4/4] Creating Statistics Dashboard...")
    create_statistics_dashboard(os.path.join(save_dir, "statistics_dashboard.png"))
    
    print("\n" + "=" * 60)
    print("   ALL VISUALIZATIONS COMPLETE!")
    print("=" * 60)
    print(f"\nCharts saved to: {save_dir}")
    print("\nGenerated files:")
    print("  - model_architecture.png")
    print("  - class_distribution.png")
    print("  - category_pie_chart.png")
    print("  - statistics_dashboard.png")


if __name__ == "__main__":
    main()
