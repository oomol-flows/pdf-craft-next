#region generated meta
import typing
class Inputs(typing.TypedDict):
    pdf_path: str
    output_path: str | None
    includes_footnotes: bool | None
    ocr_model: typing.Literal["gundam", "large", "base", "small", "tiny"] | None
    book_title: str | None
    book_authors: str | None
    table_render: typing.Literal["HTML", "CLIPPING"] | None
    latex_render: typing.Literal["MathML", "LaTeX"] | None
    generate_plot: bool | None
class Outputs(typing.TypedDict):
    epub_path: typing.NotRequired[str]
#endregion

from pathlib import Path
from oocana import Context
from pdf_craft import transform_epub, OCREventKind, TableRender, LaTeXRender, BookMeta
import torch
import math
import os

# Enable PyTorch performance optimizations for NVIDIA GPUs (RTX 3060 12GB optimized)
# These settings significantly improve GPU utilization on Ampere architecture
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for matrix operations
torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for cuDNN operations
torch.backends.cudnn.benchmark = True          # Enable cuDNN auto-tuner for optimal performance

# Additional CUDA optimizations for RTX 3060
torch.set_float32_matmul_precision('high')     # Use TF32 for float32 operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management


def apply_model_optimizations():
    """
    Apply optimizations to pdf_craft models for RTX 3060 12GB.

    Note: torch.compile() is disabled because it breaks the model's .infer() method
    which is required by pdf_craft's internal implementation. Instead, we rely on:
    - TF32 matrix operations (enabled globally)
    - cuDNN optimizations (enabled globally)
    - Efficient memory management
    """
    print(f"[PDF-to-EPUB] GPU optimizations enabled: TF32 + cuDNN benchmark (RTX 3060 12GB)")
    print(f"[PDF-to-EPUB] Note: torch.compile() disabled to preserve model compatibility")


def safe_progress_value(value):
    """Convert progress value to safe integer, replacing NaN with 0."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    return int(value)


def main(params: Inputs, context: Context) -> Outputs:
    """
    Convert PDF document to EPUB format with customizable rendering options.

    This function uses pdf-craft to transform PDF files into EPUB ebooks,
    with support for tables, LaTeX formulas, footnotes, and metadata.

    Parameters:
        params: Input parameter dictionary containing:
            - pdf_path: Path to the input PDF file
            - output_dir: Directory for output files
            - book_title: Title of the book
            - book_authors: Authors (comma-separated)
            - includes_footnotes: Whether to include footnotes
            - table_render: Table rendering mode (HTML/Markdown, optional)
            - latex_render: LaTeX rendering mode (MathML/LaTeX, optional)
            - generate_plot: Whether to generate analysis plots (optional)
        context: OOMOL context object

    Returns:
        Output dictionary containing:
            - epub_path: Path to the generated EPUB file
    """
    # Check for CUDA/GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/GPU is required for PDF conversion. "
            "This task requires an NVIDIA GPU with CUDA support. "
            "Please ensure you have a compatible GPU and CUDA drivers installed."
        )

    # Apply model optimizations for RTX 3060 12GB
    apply_model_optimizations()

    # Log GPU information with optimization settings
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print(f"[PDF-to-EPUB] Using GPU: {gpu_name} (CUDA {cuda_version}) | TF32 + cuDNN optimized (RTX 3060 12GB)")
    context.report_progress(0)

    # Convert paths to Path objects
    pdf_path = Path(params["pdf_path"])

    # Get output path, default to session directory if not provided
    output_path_param = params.get("output_path")
    if output_path_param:
        epub_path = Path(output_path_param)
        # Ensure the output path has .epub extension
        if not epub_path.suffix.lower() == ".epub":
            epub_path = epub_path.with_suffix(".epub")
    else:
        # Default to session directory with PDF filename
        session_dir = Path(context.session_dir)
        pdf_filename = pdf_path.stem
        epub_path = session_dir / f"{pdf_filename}.epub"

    # Create output directory if it doesn't exist
    output_dir = epub_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up analysis path (same directory as output)
    analysing_path = output_dir / "analysis"

    # Create models cache directory in oomol-storage
    models_cache_path = Path("/oomol-driver/oomol-storage/pdf-craft-models-cache")
    models_cache_path.mkdir(parents=True, exist_ok=True)

    # Get optional parameters with defaults
    includes_footnotes = params.get("includes_footnotes", False)  # Default: false
    generate_plot = params.get("generate_plot", False)

    # Configure table rendering
    table_render_str = params.get("table_render", "HTML")
    table_render = TableRender.HTML if table_render_str == "HTML" else TableRender.CLIPPING

    # Configure LaTeX rendering
    latex_render_str = params.get("latex_render", "MathML")
    latex_render = LaTeXRender.MATHML if latex_render_str == "MathML" else LaTeXRender.CLIPPING

    # Configure book metadata (optional)
    book_title = params.get("book_title")
    book_authors_str = params.get("book_authors")

    # Only set book metadata if title or authors are provided
    book_meta = None
    if book_title or book_authors_str:
        # Use defaults if only one is provided
        if not book_title:
            book_title = "Untitled Book"
        if not book_authors_str:
            book_authors = ["Unknown Author"]
        else:
            # Split authors by comma and strip whitespace
            book_authors = [author.strip() for author in book_authors_str.split(",") if author.strip()]
            if not book_authors:
                book_authors = ["Unknown Author"]

        book_meta = BookMeta(
            title=book_title,
            authors=book_authors
        )

    # Progress tracking callback
    def on_ocr_event(event):
        kind = OCREventKind(event.kind)
        total_pages = event.total_pages
        current_page = event.page_index + 1

        # Calculate progress percentage - safe_progress_value handles NaN/None
        progress_percent = safe_progress_value((current_page / total_pages * 100) if total_pages > 0 else 0)
        if kind == OCREventKind.START:
            if current_page == 1:
                context.report_progress(0)
                print(f"[PDF-to-EPUB] Starting conversion of {total_pages} pages")
        elif kind == OCREventKind.SKIP:
            context.report_progress(progress_percent)
            print(f"[PDF-to-EPUB] Page {current_page}/{total_pages} skipped (cached) - {progress_percent}%")
        elif kind == OCREventKind.IGNORE:
            context.report_progress(progress_percent)
            print(f"[PDF-to-EPUB] Page {current_page}/{total_pages} ignored - {progress_percent}%")
        elif kind == OCREventKind.COMPLETE:
            context.report_progress(progress_percent)
            cost_time = event.cost_time_ms / 1000
            print(f"[PDF-to-EPUB] Page {current_page}/{total_pages} completed in {cost_time:.2f}s - {progress_percent}%")
            if current_page == total_pages:
                print(f"[PDF-to-EPUB] All {total_pages} pages converted successfully!")
      

    # Get OCR model size (default to base)
    ocr_model = params.get("ocr_model") or "base"

    # Perform the conversion
    transform_epub(
        pdf_path=pdf_path,
        epub_path=epub_path,
        analysing_path=analysing_path if generate_plot else None,
        ocr_size=ocr_model,
        models_cache_path=models_cache_path,
        includes_footnotes=includes_footnotes,
        generate_plot=generate_plot,
        table_render=table_render,
        latex_render=latex_render,
        book_meta=book_meta,
        on_ocr_event=on_ocr_event
    )

    return {
        "epub_path": str(epub_path)
    }
