#!/usr/bin/env python3
"""
Generate sample figures for report testing.
This demonstrates how you can programmatically create charts and diagrams.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_sample_chart():
    """Create a simple bar chart as an example figure."""
    figures_dir = Path("DocAutomation/Figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data
    categories = ['Section A', 'Section B', 'Section C', 'Section D', 'Section E']
    values = [23, 45, 56, 78, 32]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    
    # Customize
    plt.title('Sample Data Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = figures_dir / "sample_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def create_line_plot():
    """Create a line plot showing trends."""
    figures_dir = Path("DocAutomation/Figures")
    
    # Sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='Process A', linewidth=2, color='#3498db')
    plt.plot(x, y2, label='Process B', linewidth=2, color='#e74c3c')
    
    # Customize
    plt.title('Process Comparison Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time (arbitrary units)', fontsize=12)
    plt.ylabel('Performance Metric', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_path = figures_dir / "trend_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def create_scatter_plot():
    """Create a scatter plot with correlation."""
    figures_dir = Path("DocAutomation/Figures")
    
    # Sample data
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, s=50, color='#2ecc71', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Customize
    plt.title('Correlation Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Variable X', fontsize=12)
    plt.ylabel('Variable Y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_path = figures_dir / "correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_path}")
    plt.close()


def main():
    """Generate all sample figures."""
    print("=" * 60)
    print("GENERATING SAMPLE FIGURES")
    print("=" * 60)
    
    try:
        create_sample_chart()
        create_line_plot()
        create_scatter_plot()
        print("\n" + "=" * 60)
        print("COMPLETE! Figures saved to DocAutomation/Figures/")
        print("=" * 60)
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required packages:")
        print("  pip install matplotlib numpy")


if __name__ == "__main__":
    main()
