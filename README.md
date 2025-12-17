# PDF Craft OOMOL Blocks

[中文文档](README_zh-CN.md) | English

Transform your PDF documents into modern, editable formats with ease. This OOMOL project provides powerful AI-driven conversion tools that turn PDFs into Markdown or EPUB formats, perfect for documentation, e-books, and digital content management.

## What Can This Do?

This project offers two main capabilities:

### 📄 PDF to Markdown
Convert PDF documents into clean Markdown files with all images properly extracted. Perfect for:
- Creating technical documentation
- Building static websites (Jekyll, Hugo, MkDocs)
- Version control workflows with Git
- Content management systems
- Knowledge bases and wikis

### 📚 PDF to EPUB
Transform PDFs into reflowable EPUB ebooks optimized for e-readers and mobile devices. Ideal for:
- Digital book publishing
- Academic paper distribution
- Technical manual conversion
- Mobile-friendly reading experiences
- E-reader library management

Both tools use advanced GPU-accelerated OCR technology to handle both digital PDFs and scanned documents with high accuracy.

## Key Features

- **Intelligent Text Recognition**: GPU-powered OCR handles both digital text and scanned documents
- **Image Extraction**: Automatically extracts and organizes all images with proper linking
- **Footnote Preservation**: Maintains footnotes with correct formatting and references
- **Mathematical Formula Support**: Converts LaTeX formulas to proper display formats
- **Table Handling**: Preserves complex table structures with multiple rendering options
- **Flexible OCR Models**: Choose from 5 model sizes to balance speed and accuracy
- **GPU Optimization**: Adjustable memory usage and precision settings for optimal performance
- **Analysis Visualizations**: Optional diagnostic charts for quality assurance
- **Metadata Customization**: Set book titles, authors, and rendering preferences for EPUB output

## Available Blocks

### PDF to Markdown Block

Converts PDF documents to Markdown format with automatic image extraction and organization.

**What You Need to Provide:**
- **PDF File**: The PDF document you want to convert
- **Output Location** (optional): Where to save the Markdown file and images (defaults to session directory)

**Optional Settings:**
- **Include Footnotes**: Keep or remove footnotes from the output (default: excluded)
- **OCR Model Size**: Choose accuracy vs. speed (options: tiny, small, gundam, base, large; default: base)
- **Generate Analysis**: Create diagnostic charts showing conversion quality and performance
- **Optimization Level**: Select GPU precision strategy (balanced or quality; default: balanced)
- **GPU Memory**: Control how much GPU memory to use (10-100%; default: 90%)

**What You Get:**
- A ZIP archive containing:
  - Clean Markdown file with formatted text, headers, lists, tables, and formulas
  - Images folder with all extracted graphics in original quality
  - Properly linked references between Markdown and images

**Best For:** Documentation systems, static sites, Git-based workflows, content editing

---

### PDF to EPUB Block

Converts PDF documents to EPUB ebook format suitable for e-readers and mobile devices.

**What You Need to Provide:**
- **PDF File**: The PDF document you want to convert
- **Output Location** (optional): Where to save the EPUB file (defaults to session directory)

**Optional Settings:**
- **Book Title**: Set the title metadata for e-reader display
- **Book Authors**: Specify author names (comma-separated for multiple authors)
- **Include Footnotes**: Preserve or remove footnotes (default: excluded)
- **OCR Model Size**: Choose processing quality (tiny, small, gundam, base, large; default: base)
- **Table Rendering**: How to display tables (HTML or Markdown; default: HTML)
- **LaTeX Rendering**: How to show mathematical formulas (MathML or LaTeX; default: MathML)
- **Generate Analysis**: Create conversion quality visualizations

**What You Get:**
- A fully formatted EPUB file ready to open in any e-reader app
- Properly structured chapters and navigation
- Embedded metadata for library organization
- Reflowable text that adapts to screen sizes

**Best For:** E-books, academic papers, digital publishing, mobile reading

## Getting Started

### Prerequisites

To use these blocks, you need:
- An NVIDIA GPU with CUDA support (required for OCR processing)
- Sufficient GPU memory (at least 6-8GB VRAM recommended)
- Disk space for model caching (approximately 3-5GB)

### Installation

The project handles installation automatically when you first load it in OOMOL:

1. **System Dependencies**: Installs poppler-utils for PDF processing
2. **Python Environment**: Sets up PyTorch with CUDA 12.x support
3. **AI Models**: Downloads and caches OCR models on first use
4. **Python Libraries**: Installs all required packages via poetry

No manual setup required - just load the project and start converting!

### Using the Blocks

#### In OOMOL Flows

Reference these blocks in your workflows using:
- `self::pdf-to-markdown` for Markdown conversion
- `self::pdf-to-epub` for EPUB conversion

**Example Flow:**

```yaml
nodes:
  - node_id: convert-to-markdown#1
    task: self::pdf-to-markdown
    inputs_from:
      - handle: pdf_path
        value: "/path/to/your/document.pdf"
      - handle: output_path
        value: "/oomol-driver/oomol-storage/output/document.md"
      - handle: includes_footnotes
        value: true
      - handle: ocr_size
        value: "base"

  - node_id: convert-to-epub#1
    task: self::pdf-to-epub
    inputs_from:
      - handle: pdf_path
        value: "/path/to/your/document.pdf"
      - handle: output_path
        value: "/oomol-driver/oomol-storage/output/book.epub"
      - handle: book_title
        value: "My Technical Book"
      - handle: book_authors
        value: "Jane Doe, John Smith"
      - handle: table_render
        value: "HTML"
      - handle: latex_render
        value: "MathML"
```

#### Test Flow

A sample test flow is available at [flows/test-pdf-conversion/flow.oo.yaml](flows/test-pdf-conversion/flow.oo.yaml). Update the `pdf_path` values to point to your own PDF files.

## Choosing the Right OCR Model

Different OCR models offer trade-offs between processing speed and accuracy:

| Model Size | Speed | Quality | GPU Memory | Best For |
|------------|-------|---------|------------|----------|
| **tiny** | Fastest | Lowest | ~2GB | High-quality PDFs with simple layouts |
| **small** | Fast | Good | ~4GB | Standard documents, good scan quality |
| **gundam** | Balanced | Very Good | ~6GB | General purpose, recommended for most users |
| **base** | Moderate | High | ~8GB | Default option, excellent accuracy (recommended) |
| **large** | Slowest | Highest | ~12GB | Complex layouts, poor scans, maximum accuracy |

**Recommendation**: Start with **base** (default) for best results. Use **gundam** for faster processing with good quality, or **large** for challenging documents.

## GPU Optimization Guide

### Optimization Levels

- **Balanced** (default): Uses bfloat16 precision for optimal speed/quality balance on modern GPUs (RTX 30/40 series)
- **Quality**: Uses float16 precision for slightly higher accuracy, may be slower

### GPU Memory Management

The `gpu_memory_fraction` setting controls how much GPU memory to allocate:

- **0.9 (90%, default)**: Maximum performance for dedicated processing
- **0.7 (70%)**: Good balance, leaves memory for other applications
- **0.5 (50%)**: Conservative, suitable for shared GPU environments

**Tip**: If you encounter out-of-memory errors, reduce this value or use a smaller OCR model.

## Storage Locations

- **Model Cache**: `/oomol-driver/oomol-storage/pdf-craft-models-cache` (automatically managed)
- **Output Files**: Recommended to use paths under `/oomol-driver/oomol-storage/` for runtime files
- **Session Files**: Default outputs go to session-specific directories when no output path is specified

## Technical Details

### System Requirements

- Python 3.10-3.12
- NVIDIA GPU with CUDA 12.x support
- PyTorch 2.9.0 with CUDA acceleration
- Poppler utilities for PDF processing

### Main Dependencies

- [pdf-craft](https://github.com/oomol-lab/pdf-craft): Core PDF conversion library
- PyTorch: Deep learning framework for OCR models
- Transformers: Hugging Face model support
- PyMuPDF: PDF parsing and manipulation
- EbookLib: EPUB generation

### Architecture

Both blocks use a GPU-accelerated pipeline:
1. **PDF Parsing**: Extract pages and basic structure
2. **OCR Processing**: AI-powered text recognition with configurable models
3. **Image Extraction**: Automatic image detection and export
4. **Content Assembly**: Reconstruct document structure in target format
5. **Format Generation**: Create final Markdown or EPUB output

### Performance Characteristics

- **Processing Speed**: Approximately 1-3 seconds per page (varies by OCR model and GPU)
- **GPU Utilization**: Typically 70-95% during OCR processing
- **Memory Usage**: 4-12GB GPU memory depending on model size
- **Accuracy**: 95-99% text recognition accuracy for most documents

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Solution: Reduce `gpu_memory_fraction` to 0.7 or 0.5
- Alternative: Use a smaller OCR model (small or gundam)

**Low Quality Output**
- Solution: Switch to a larger OCR model (base or large)
- Alternative: Enable analysis plots to diagnose quality issues

**Slow Processing**
- Solution: Use a smaller OCR model (gundam or small)
- Alternative: Ensure GPU optimization level is set to "balanced"

**Missing Footnotes**
- Solution: Set `includes_footnotes` to `true`

**Incorrect Table Formatting (EPUB)**
- Solution: Try switching between HTML and Markdown rendering modes

## Contributing

This project is built on [pdf-craft](https://github.com/oomol-lab/pdf-craft). For issues or contributions related to the core conversion engine, please visit the pdf-craft repository.

## License

Please refer to the [pdf-craft license](https://github.com/oomol-lab/pdf-craft) for usage terms and conditions.

---

**Need Help?** Check the [OOMOL documentation](https://github.com/oomol-flows/pdf-craft-next) or open an issue on GitHub.
