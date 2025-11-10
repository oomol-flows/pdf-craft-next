#region generated meta
import typing
class Inputs(typing.TypedDict):
    pdf_path: str
    output_dir: str
    book_title: str
    book_authors: str
    includes_footnotes: bool
    table_render: typing.Literal["HTML", "Markdown"] | None
    latex_render: typing.Literal["MathML", "LaTeX"] | None
    generate_plot: bool | None
    ocr_model: typing.Literal["gundam", "large", "base", "small", "tiny"] | None
class Outputs(typing.TypedDict):
    epub_path: typing.NotRequired[str]
#endregion

from pathlib import Path
from oocana import Context
from pdf_craft import transform_epub, OCREventKind, TableRender, LaTeXRender, BookMeta
import torch
import math


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

    # Enable GPU performance optimizations
    # Enable cuDNN benchmark mode for automatic algorithm selection
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for faster computation on Ampere GPUs (RTX 30xx/40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set memory allocation strategy for better GPU utilization
    torch.cuda.empty_cache()

    # Log GPU information
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    context.report_progress({
        "progress": 0,
        "message": f"Using GPU: {gpu_name} (CUDA {cuda_version}) with TF32 and cuDNN optimizations enabled"
    })

    # Convert paths to Path objects
    pdf_path = Path(params["pdf_path"])
    output_dir = Path(params["output_dir"])

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get PDF filename without extension for EPUB output
    pdf_filename = pdf_path.stem
    epub_path = output_dir / f"{pdf_filename}.epub"

    # Set up analysis path
    analysing_path = output_dir / "analysis"

    # Create models cache directory in oomol-storage
    models_cache_path = Path("/oomol-driver/oomol-storage/pdf-craft-models-cache")
    models_cache_path.mkdir(parents=True, exist_ok=True)

    # Get optional parameters
    includes_footnotes = params.get("includes_footnotes", True)
    generate_plot = params.get("generate_plot", False)

    # Configure table rendering
    table_render_str = params.get("table_render", "HTML")
    table_render = TableRender.HTML if table_render_str == "HTML" else TableRender.MARKDOWN

    # Configure LaTeX rendering
    latex_render_str = params.get("latex_render", "MathML")
    latex_render = LaTeXRender.MATHML if latex_render_str == "MathML" else LaTeXRender.LATEX

    # Configure book metadata
    book_title = params.get("book_title", "Untitled Book")
    book_authors_str = params.get("book_authors", "Unknown Author")
    # Split authors by comma and strip whitespace
    book_authors = [author.strip() for author in book_authors_str.split(",")]

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

    # Get OCR model size (default to gundam for best quality)
    ocr_model = params.get("ocr_model", "gundam")

    # Perform the conversion
    transform_epub(
        pdf_path=pdf_path,
        epub_path=epub_path,
        analysing_path=analysing_path if generate_plot else None,
        model=ocr_model,
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
