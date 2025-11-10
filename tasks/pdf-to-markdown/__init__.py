#region generated meta
import typing
class Inputs(typing.TypedDict):
    pdf_path: str
    output_dir: str
    includes_footnotes: bool
    generate_plot: bool | None
class Outputs(typing.TypedDict):
    markdown_path: typing.NotRequired[str]
    assets_dir: typing.NotRequired[str]
#endregion

from pathlib import Path
from oocana import Context
from pdf_craft import transform_markdown, OCREventKind


def main(params: Inputs, context: Context) -> Outputs:
    """
    Convert PDF document to Markdown format with image extraction.

    This function uses pdf-craft to transform PDF files into Markdown,
    extracting images and formatting text. It supports footnotes and
    optional analysis visualization.

    Parameters:
        params: Input parameter dictionary containing:
            - pdf_path: Path to the input PDF file
            - output_dir: Directory for output files
            - includes_footnotes: Whether to include footnotes
            - generate_plot: Whether to generate analysis plots (optional)
        context: OOMOL context object

    Returns:
        Output dictionary containing:
            - markdown_path: Path to the generated markdown file
            - assets_dir: Directory containing extracted images
    """
    # Convert paths to Path objects
    pdf_path = Path(params["pdf_path"])
    output_dir = Path(params["output_dir"])

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output paths
    markdown_path = output_dir / "output.md"
    assets_dir = output_dir / "images"
    analysing_path = output_dir / "analysis"

    # Create models cache directory in oomol-storage
    models_cache_path = Path("/oomol-driver/oomol-storage/pdf-craft-models-cache")
    models_cache_path.mkdir(parents=True, exist_ok=True)

    # Get optional parameters
    includes_footnotes = params.get("includes_footnotes", True)
    generate_plot = params.get("generate_plot", False)

    # Progress tracking callback
    def on_ocr_event(event):
        kind = OCREventKind(event.kind)
        if kind == OCREventKind.START:
            context.report_progress(0, "Starting OCR processing...")
        elif kind == OCREventKind.PROGRESS:
            total_pages = event.total or 1
            current_page = event.current or 0
            progress = (current_page / total_pages) * 100
            context.report_progress(progress, f"Processing page {current_page}/{total_pages}")
        elif kind == OCREventKind.COMPLETE:
            context.report_progress(100, "PDF to Markdown conversion completed")

    # Perform the conversion
    transform_markdown(
        pdf_path=pdf_path,
        markdown_path=markdown_path,
        markdown_assets_path=Path("images"),
        analysing_path=analysing_path if generate_plot else None,
        models_cache_path=models_cache_path,
        includes_footnotes=includes_footnotes,
        generate_plot=generate_plot,
        on_ocr_event=on_ocr_event
    )

    return {
        "markdown_path": str(markdown_path),
        "assets_dir": str(assets_dir)
    }
