#!/bin/bash

# Report Generation Script
# Compiles markdown files and generates reports in multiple formats

echo "=================================="
echo "Report Generation Automation"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if pandoc is available
if ! command -v pandoc &> /dev/null; then
    echo "Warning: Pandoc is not installed. PDF/DOCX conversion will fail."
    echo "Install with: sudo apt-get install pandoc texlive-xetex (Ubuntu)"
    echo "            or: brew install pandoc basictex (macOS)"
    echo ""
fi

# Create directory structure if it doesn't exist
mkdir -p DocAutomation/{MD_Main,Figures,Out}

# Copy templates if MD_Main is empty
if [ -z "$(ls -A DocAutomation/MD_Main 2>/dev/null)" ]; then
    echo "Setting up template files..."
    cp 0*.md DocAutomation/MD_Main/ 2>/dev/null || echo "Template files not found in current directory"
fi

# Copy config if it doesn't exist
if [ ! -f "DocAutomation/report_config.yaml" ]; then
    echo "Setting up configuration..."
    cp report_config.yaml DocAutomation/ 2>/dev/null || echo "Config file not found"
fi

# Run the Python script
echo "Running report compiler..."
python3 compile_report.py

echo ""
echo "=================================="
echo "Done!"
echo "=================================="
echo ""
echo "Check DocAutomation/Out/ for generated reports"
