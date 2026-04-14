#!/usr/bin/env python3
"""
Report Generation Automation Script
Compiles markdown files from MD_Main into a single report and converts to various formats.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import yaml
import re


class ReportCompiler:
    def __init__(self, base_dir="DocAutomation"):
        self.base_dir = Path(base_dir)
        self.md_main = self.base_dir / "MD_Main"
        self.figures = self.base_dir / "Figures"
        self.out = self.base_dir / "Out"
        self.config_file = self.base_dir / "report_config.yaml"
        
        # Create directories if they don't exist
        self.md_main.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Load report configuration from YAML file."""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'sections': [
                    '01_title.md',
                    '02_abstract.md',
                    '03_introduction.md',
                    '04_methodology.md',
                    '05_results.md',
                    '06_discussion.md',
                    '07_conclusion.md',
                    '08_references.md'
                ],
                'metadata': {
                    'title': 'Report Title',
                    'author': 'Author Name',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
            }
    
    def generate_dynamic_content(self):
        """
        Placeholder for generating dynamic MD sections.
        This will be implemented later to create specific report parts programmatically.
        """
        # TODO: Add logic to generate dynamic content
        # For example:
        # - Data analysis sections
        # - Automated charts/tables
        # - Calculated results
        pass
    
    def compile_markdown(self, output_name="report.md"):
        """Compile all markdown sections into a single file."""
        config = self.load_config()
        output_path = self.out / output_name
        
        print(f"Compiling report to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Write YAML frontmatter if metadata exists
            #if 'metadata' in config:
            #    outfile.write('---\n')
            #    yaml.dump(config['metadata'], outfile, default_flow_style=False)
            #    outfile.write('---\n\n')
            
            # Compile sections in order
            for section_file in config['sections']:
                section_path = self.md_main / section_file
                
                if section_path.exists():
                    print(f"  Adding: {section_file}")
                    with open(section_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        # Fix relative image paths to point to Figures directory
                        content = self._fix_image_paths(content)
                        outfile.write(content)
                        outfile.write('\n\n')
                else:
                    print(f"  Warning: {section_file} not found, skipping")
        
        print(f"✓ Compiled markdown saved to: {output_path}")
        return output_path
    
    def _fix_image_paths(self, content):
        """Convert relative image paths to absolute paths for pandoc."""
        # Pattern to match markdown images: ![alt](path)
        def replace_path(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # If it's already an absolute path or URL, leave it
            if img_path.startswith(('http://', 'https://', '/')):
                return match.group(0)
            
            # Convert to absolute path pointing to Figures directory
            abs_path = self.figures / img_path
            return f'![{alt_text}]({abs_path})'
        
        return re.sub(r'!\[(.*?)\]\((.*?)\)', replace_path, content)
    
    def convert_to_pdf(self, md_file="report.md", pdf_name="report.pdf"):
        """Convert markdown to PDF using pandoc."""
        md_path = self.out / md_file
        pdf_path = self.out / pdf_name
        
        if not md_path.exists():
            print(f"Error: {md_path} does not exist")
            return False
        
        print(f"\nConverting to PDF: {pdf_path}")
        portable_xelatex = r"C:\MatlabR15\Work\miktex_24.1-Portable\texmfs\install\miktex\bin\x64\xelatex.exe"

        cmd = [
            'pandoc',
            str(md_path),
            '-o', str(pdf_path),
            f'--pdf-engine={portable_xelatex}',  # Use portable version
            '--toc',
            '--number-sections'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
            # Filter output - only show real errors, not MiKTeX nags
            if result.stderr:
                lines = result.stderr.split('\n')
                errors = [line for line in lines 
                        if line and 
                        'major issue: So far, you have not checked' not in line and
                        'Missing character' not in line]
                if errors:
                    print('\n'.join(errors))
            
            print(f"✓ PDF created: {pdf_path}")
            os.startfile(pdf_path)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error converting to PDF: {e}")
            if e.stderr:
                print(e.stderr)
            return False
    
    def convert_to_docx(self, md_file="report.md", docx_name="report.docx"):
        """Convert markdown to DOCX using pandoc."""
        md_path = self.out / md_file
        docx_path = self.out / docx_name
        
        if not md_path.exists():
            print(f"Error: {md_path} does not exist")
            return False
        
        print(f"\nConverting to DOCX: {docx_path}")
        
        cmd = [
            'pandoc',
            str(md_path),
            '-o', str(docx_path),
            '--toc',
            '--number-sections'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ DOCX created: {docx_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error converting to DOCX: {e}")
            return False
        except FileNotFoundError:
            print("Error: pandoc not found. Please install pandoc first.")
            return False
    
    def compile_and_convert(self, formats=['pdf', 'docx']):
        """Main workflow: compile markdown and convert to specified formats."""
        print("=" * 60)
        print("REPORT GENERATION AUTOMATION")
        print("=" * 60)
        
        # Generate any dynamic content first
        self.generate_dynamic_content()
        
        # Compile markdown
        md_path = self.compile_markdown()
        
        # Convert to requested formats
        for fmt in formats:
            if fmt.lower() == 'pdf':
                self.convert_to_pdf()
            elif fmt.lower() == 'docx':
                self.convert_to_docx()
            elif fmt.lower() == 'html':
                # Can add HTML conversion later
                pass
        
        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)


def main():
    compiler = ReportCompiler()
    
    # Compile and convert to both PDF and DOCX
    compiler.compile_and_convert(formats=['pdf', 'docx'])


if __name__ == "__main__":
    main()
