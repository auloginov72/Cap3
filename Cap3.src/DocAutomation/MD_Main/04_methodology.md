# Methodology

## System Architecture

The report generation system consists of three main components:

1. **Content Layer** - Markdown files in MD_Main directory
2. **Processing Layer** - Python compilation script
3. **Output Layer** - Generated reports in multiple formats

## Technical Stack

The following tools and technologies are used:

| Component | Tool | Purpose |
|-----------|------|---------|
| Content Format | Markdown | Human-readable source format |
| Compilation | Python 3.x | Orchestration and automation |
| Conversion | Pandoc | Multi-format document conversion |
| Configuration | YAML | Structured settings management |

## Directory Structure

```
DocAutomation/
├── MD_Main/          # Source markdown files
│   ├── 01_title.md
│   ├── 02_abstract.md
│   └── ...
├── Figures/          # Images and diagrams
│   └── example.png
├── Out/              # Generated outputs
│   ├── report.md
│   ├── report.pdf
│   └── report.docx
└── report_config.yaml
```

## Compilation Process

The compilation process follows these steps:

```python
# Pseudocode for report compilation
def compile_report():
    1. Load configuration from YAML
    2. Generate any dynamic content
    3. Collect markdown sections in order
    4. Fix image paths for absolute references
    5. Compile into single markdown file
    6. Convert to target formats using pandoc
```

## Image Integration

Images are stored in the `Figures/` directory and referenced in markdown as:

```markdown
![Image Description](example.png)
```

The system automatically converts relative paths to absolute paths during compilation.

## Configuration Management

Report structure is defined in `report_config.yaml`:

- Section order
- Metadata (title, author, date)
- Pandoc conversion options

This allows for easy customization without modifying the Python code.
