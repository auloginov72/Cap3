#!/usr/bin/env python3
"""
Setup script for the report automation project.
Creates the proper directory structure and copies template files.
"""

import os
import shutil
from pathlib import Path


def setup_project():
    """Create the complete project structure."""
    
    print("=" * 60)
    print("SETTING UP REPORT AUTOMATION PROJECT")
    print("=" * 60)
    print()
    
    # Define base directory
    base_dir = Path("DocAutomation")
    
    # Create directory structure
    directories = [
        base_dir / "MD_Main",
        base_dir / "MD_Templates",  # For reusable templates
        base_dir / "Figures",
        base_dir / "Data",          # For data files
        base_dir / "Out",
        base_dir / "archive",       # For old versions
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print()
    print("Copying template files to MD_Main...")
    
    # Copy markdown templates
    template_files = [
        "01_title.md",
        "02_abstract.md", 
        "03_introduction.md",
        "04_methodology.md",
        "05_results.md",
        "06_discussion.md",
        "07_conclusion.md",
        "08_references.md"
    ]
    
    for template in template_files:
        src = Path(template)
        dst = base_dir / "MD_Main" / template
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"  ✓ {template}")
        elif dst.exists():
            print(f"  - {template} (already exists)")
        else:
            print(f"  ✗ {template} (source not found)")
    
    print()
    print("Copying configuration file...")
    
    # Copy config file
    config_src = Path("report_config.yaml")
    config_dst = base_dir / "report_config.yaml"
    if config_src.exists() and not config_dst.exists():
        shutil.copy(config_src, config_dst)
        print(f"  ✓ report_config.yaml")
    elif config_dst.exists():
        print(f"  - report_config.yaml (already exists)")
    else:
        print(f"  ✗ report_config.yaml (source not found)")
    
    print()
    print("Creating README in DocAutomation...")
    
    # Create a simple README in DocAutomation
    readme_content = """# DocAutomation

This directory contains the report generation system.

## Quick Start

1. Edit markdown files in `MD_Main/`
2. Add images to `Figures/`
3. Run: `python3 ../compile_report.py`
4. Find outputs in `Out/`

## Directory Structure

- **MD_Main/** - Source markdown sections
- **MD_Templates/** - Reusable section templates
- **Figures/** - Images and diagrams
- **Data/** - Data files for dynamic content
- **Out/** - Generated reports
- **archive/** - Old versions

## Configuration

Edit `report_config.yaml` to customize:
- Section order
- Metadata (title, author, etc.)
- Pandoc options
"""
    
    readme_path = base_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print("  ✓ README.md created")
    else:
        print("  - README.md (already exists)")
    
    print()
    print("=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Project structure:")
    print_tree(base_dir)
    print()
    print("Next steps:")
    print("  1. Review templates in DocAutomation/MD_Main/")
    print("  2. Edit content as needed")
    print("  3. Run: python3 compile_report.py")
    print("  4. Check DocAutomation/Out/ for results")
    print()


def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Print a tree view of the directory structure."""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass


if __name__ == "__main__":
    setup_project()
