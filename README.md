# PDF Craft OOMOL Blocks

This OOMOL project provides task blocks for converting PDF documents to Markdown and EPUB formats using [pdf-craft](https://github.com/oomol-lab/pdf-craft).

## Features

- **PDF to Markdown**: Convert PDF documents to Markdown format with image extraction
- **PDF to EPUB**: Convert PDF documents to EPUB ebooks with customizable rendering options
- Support for footnotes, tables, and LaTeX formulas
- Progress tracking during conversion
- Configurable output options

## Task Blocks

### PDF to Markdown

Converts PDF documents to Markdown format with automatic image extraction.

**Inputs:**
- `pdf_path` (required): Path to the input PDF file
- `output_dir` (required): Directory where the markdown file and assets will be saved
- `includes_footnotes` (required, default: true): Whether to include footnotes in the output
- `generate_plot` (optional, default: false): Whether to generate analysis visualizations

**Outputs:**
- `markdown_path`: Path to the generated markdown file
- `assets_dir`: Directory containing the extracted images and assets

### PDF to EPUB

Converts PDF documents to EPUB ebook format with metadata and rendering customization.

**Inputs:**
- `pdf_path` (required): Path to the input PDF file
- `output_dir` (required): Directory where the EPUB file will be saved
- `book_title` (required, default: "Untitled Book"): Title of the book for EPUB metadata
- `book_authors` (required, default: "Unknown Author"): Authors (comma-separated for multiple)
- `includes_footnotes` (required, default: true): Whether to include footnotes
- `table_render` (optional, default: "HTML"): How to render tables (HTML or Markdown)
- `latex_render` (optional, default: "MathML"): How to render LaTeX formulas (MathML or LaTeX)
- `generate_plot` (optional, default: false): Whether to generate analysis visualizations

**Outputs:**
- `epub_path`: Path to the generated EPUB file

## Installation

This project uses poetry for Python dependency management. The pdf-craft library is installed from the GitHub `next` branch.

### Automatic Installation

Dependencies are automatically installed when the container is first loaded via the bootstrap script in [package.oo.yaml](package.oo.yaml). The bootstrap process:

1. Installs Node.js dependencies via `npm install`
2. Installs PyTorch with CUDA 12.6 support via pip
3. Installs all Python dependencies including pdf-craft via `poetry install`

### PyTorch and CUDA

**Important:** This project requires PyTorch with CUDA support to be installed **before** pdf-craft. The bootstrap script handles this automatically by installing:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

This installs PyTorch 2.9+ with CUDA 12.6 support, which is compatible with CUDA 12.1-12.9 environments.

## Usage

### Using in Flows

You can use these blocks in your OOMOL flows by referencing them with `self::pdf-to-markdown` or `self::pdf-to-epub`.

Example flow configuration:

```yaml
nodes:
  - node_id: pdf-to-markdown#1
    task: self::pdf-to-markdown
    inputs_from:
      - handle: pdf_path
        value: "/path/to/document.pdf"
      - handle: output_dir
        value: "/oomol-driver/oomol-storage/markdown-output"
      - handle: includes_footnotes
        value: true

  - node_id: pdf-to-epub#1
    task: self::pdf-to-epub
    inputs_from:
      - handle: pdf_path
        value: "/path/to/document.pdf"
      - handle: output_dir
        value: "/oomol-driver/oomol-storage/epub-output"
      - handle: book_title
        value: "My Book"
      - handle: book_authors
        value: "John Doe, Jane Smith"
      - handle: table_render
        value: "HTML"
      - handle: latex_render
        value: "MathML"
```

### Test Flow

A test flow is available at [flows/test-pdf-conversion/flow.oo.yaml](flows/test-pdf-conversion/flow.oo.yaml). Update the `pdf_path` values to point to your test PDF file.

## Storage

- Model cache: Models are automatically downloaded and cached in `/oomol-driver/oomol-storage/pdf-craft-models-cache`
- Output files: Recommended to use paths under `/oomol-driver/oomol-storage/` for runtime generated files

## Requirements

- Python 3.10-3.12
- NVIDIA GPU with CUDA support (the pdf-craft library requires GPU acceleration)
- Sufficient disk space for model caching (several GB)

## Dependencies

Main dependency:
- [pdf-craft](https://github.com/oomol-lab/pdf-craft) (version 1.0.0rc1 from next branch)

Additional dependencies are managed automatically through poetry and include:
- PyTorch with CUDA support
- transformers
- matplotlib
- pymupdf
- And other required libraries

## License

Please refer to the [pdf-craft license](https://github.com/oomol-lab/pdf-craft) for usage terms.
