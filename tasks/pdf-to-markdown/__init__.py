#region generated meta
import typing
class Inputs(typing.TypedDict):
    pdf_path: str
    output_dir: str
    includes_footnotes: bool
    generate_plot: bool | None
    optimization_level: typing.Literal["balanced", "speed", "quality"] | None
    gpu_memory_fraction: float | None
class Outputs(typing.TypedDict):
    markdown_path: typing.NotRequired[str]
    assets_dir: typing.NotRequired[str]
#endregion

from pathlib import Path
from oocana import Context
import torch
import math
import os

from pdf_craft import transform_markdown, OCREventKind

# Enable PyTorch performance optimizations for NVIDIA GPUs
# These settings significantly improve GPU utilization on Ampere and later architectures (RTX 30/40 series)
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for matrix operations
torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for cuDNN operations
torch.backends.cudnn.benchmark = True          # Enable cuDNN auto-tuner for optimal performance

# Additional CUDA optimizations for RTX 4090
torch.set_float32_matmul_precision('high')     # Use TF32 for float32 operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management


def setup_optimization(optimization_level: str, gpu_memory_fraction: float):
    """
    Configure GPU optimization settings based on the specified level.

    Parameters:
        optimization_level: One of 'balanced', 'speed', or 'quality'
        gpu_memory_fraction: Maximum fraction of GPU memory to use (0.1-1.0)

    Returns:
        dict: Configuration settings for pdf_craft
    """
    # Set GPU memory fraction
    if gpu_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)

    config = {
        "dtype": torch.float16,  # Default
        "use_flash_attention": False,
    }

    if optimization_level == "quality":
        # Conservative mode: float16
        config["dtype"] = torch.float16
        config["use_flash_attention"] = False

    elif optimization_level == "balanced":
        # Recommended: bfloat16 for better numerical stability
        config["dtype"] = torch.bfloat16
        config["use_flash_attention"] = False

    elif optimization_level == "speed":
        # Maximum speed: bfloat16 + Flash Attention 2
        config["dtype"] = torch.bfloat16
        config["use_flash_attention"] = True

    return config


def apply_model_optimizations(config: dict):
    """
    Apply optimizations to pdf_craft models via monkey patching.

    This function patches the model loading process to apply dtype conversion
    and Flash Attention 2 optimizations. Note: torch.compile has been removed
    due to compatibility issues with the OOMOL task execution environment.

    Parameters:
        config: Configuration dict from setup_optimization()
    """
    from doc_page_extractor.model import DeepSeekOCRHugginfaceModel
    from transformers import AutoModel
    import torch

    # Store original _ensure_models method
    original_ensure_models = DeepSeekOCRHugginfaceModel._ensure_models

    def optimized_ensure_models(self):
        # Call original method to load models
        result = original_ensure_models(self)

        # Apply optimizations to each model after loading
        target_dtype = config["dtype"]
        print(f"[PDF-to-Markdown] Applying dtype optimization: {target_dtype}")

        for idx, model in enumerate(result.llms):
            # Model is already on CUDA and in bfloat16, may need to convert
            if target_dtype != torch.bfloat16:
                result.llms[idx] = model.to(dtype=target_dtype)
                print(f"[PDF-to-Markdown] Model {idx} converted to {target_dtype}")
            else:
                print(f"[PDF-to-Markdown] Model {idx} using {target_dtype} (no conversion needed)")

        # Flash Attention is controlled by _ATTN_IMPLEMENTATION in model.py
        if config["use_flash_attention"]:
            print("[PDF-to-Markdown] Flash Attention 2 is automatically enabled by doc_page_extractor")

        return result

    # Apply the monkey patch
    DeepSeekOCRHugginfaceModel._ensure_models = optimized_ensure_models


def safe_progress_value(value):
    """Convert progress value to safe integer, replacing NaN with 0."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    return int(value)


def log_progress_report(progress_percent: int, message: str, business_context: str = ""):
    """
    Log progress report for debugging and issue tracking.

    Parameters:
        progress_percent: Progress percentage (0-100)
        message: Progress message to display
        business_context: Additional business context for debugging
    """
    import time

    print(f"[PDF-to-Markdown] PROGRESS UPDATE: {progress_percent}% | MESSAGE: {message}")
    if business_context:
        print(f"[PDF-to-Markdown] BUSINESS CONTEXT: {business_context}")

    # Show timing information
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    gpu_mode = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU Mode"
    print(f"[PDF-to-Markdown] TIMESTAMP: {timestamp} | MODE: {gpu_mode}")


def main(params: Inputs, context: Context) -> Outputs:
    """
    Convert PDF document to Markdown format with image extraction.

    This function uses pdf-craft to transform PDF files into Markdown,
    extracting images and formatting text. It supports footnotes and
    optional analysis visualization with GPU acceleration optimizations.

    Parameters:
        params: Input parameter dictionary containing:
            - pdf_path: Path to the input PDF file
            - output_dir: Directory for output files
            - includes_footnotes: Whether to include footnotes
            - generate_plot: Whether to generate analysis plots (optional)
            - optimization_level: GPU optimization level (optional)
            - gpu_memory_fraction: GPU memory limit (optional)
        context: OOMOL context object

    Returns:
        Output dictionary containing:
            - markdown_path: Path to the generated markdown file
            - assets_dir: Directory containing extracted images
    """
    # Check for CUDA/GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/GPU is required for PDF conversion. "
            "This task requires an NVIDIA GPU with CUDA support. "
            "Please ensure you have a compatible GPU and CUDA drivers installed."
        )

    # Get optimization parameters
    optimization_level = params.get("optimization_level", "balanced")
    gpu_memory_fraction = params.get("gpu_memory_fraction", 0.9)

    # Setup GPU optimizations
    opt_config = setup_optimization(optimization_level, gpu_memory_fraction)

    # Apply model optimizations via monkey patching
    apply_model_optimizations(opt_config)

    # Log GPU information with optimization settings
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    dtype_name = "bfloat16" if opt_config["dtype"] == torch.bfloat16 else "float16"
    gpu_message = f"Using GPU: {gpu_name} (CUDA {cuda_version}) | Optimization: {optimization_level} ({dtype_name})"

    if opt_config["use_flash_attention"]:
        gpu_message += " + Flash Attention 2"

    # Log progress with business context
    opt_context = f"GPU memory limit: {gpu_memory_fraction*100:.0f}% | Dtype: {dtype_name} | Flash-Attn: {opt_config['use_flash_attention']}"
    log_progress_report(0, gpu_message, opt_context)
    context.report_progress(0)

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

    # Log conversion start with business context
    start_message = f"Initializing PDF-to-Markdown conversion"
    start_context = f"PDF: {pdf_path.name} | Output: {output_dir} | Footnotes: {includes_footnotes} | Plots: {generate_plot}"
    log_progress_report(0, start_message, start_context)
    print(f"[PDF-to-Markdown] {start_message}")

    # Progress tracking callback
    def on_ocr_event(event):
        kind = OCREventKind(event.kind)
        total_pages = event.total_pages
        current_page = event.page_index + 1

        # Calculate progress percentage - safe_progress_value handles NaN/None
        progress_percent = safe_progress_value((current_page / total_pages * 100) if total_pages > 0 else 0)
        if kind == OCREventKind.START:
            if current_page == 1:
                message = f"Starting conversion of {total_pages} pages"
                business_context = f"OCR initialization - processing PDF with {total_pages} total pages"

                # Log detailed progress information
                log_progress_report(0, message, business_context)
                context.report_progress(0)
                print(f"[PDF-to-Markdown] {message}")

        elif kind == OCREventKind.SKIP:
            message = f"Page {current_page}/{total_pages} skipped (cached)"
            business_context = f"OCR cache hit - page {current_page} already processed, skipping GPU computation"

            # Log detailed progress information
            log_progress_report(progress_percent, message, business_context)
            context.report_progress(progress_percent)
            print(f"[PDF-to-Markdown] {message} - {progress_percent}%")

        elif kind == OCREventKind.IGNORE:
            message = f"Page {current_page}/{total_pages} ignored"
            business_context = f"OCR ignored - page {current_page} marked for ignoring, likely due to format or content"

            # Log detailed progress information
            log_progress_report(progress_percent, message, business_context)
            context.report_progress(progress_percent)
            print(f"[PDF-to-Markdown] {message} - {progress_percent}%")

        elif kind == OCREventKind.COMPLETE:
            cost_time = event.cost_time_ms / 1000
            message = f"Page {current_page}/{total_pages} completed in {cost_time:.2f}s"
            business_context = f"OCR processing complete - page {current_page} successfully converted, GPU time: {cost_time:.2f}s"

            # Log detailed progress information
            log_progress_report(progress_percent, message, business_context)
            context.report_progress(progress_percent)
            print(f"[PDF-to-Markdown] {message} - {progress_percent}%")

            if current_page == total_pages:
                complete_message = f"All {total_pages} pages converted successfully!"
                complete_context = f"PDF-to-Markdown conversion complete - total {total_pages} pages processed, GPU utilization optimized"

                # Final completion log
                log_progress_report(100, complete_message, complete_context)
                print(f"[PDF-to-Markdown] {complete_message}")

    # Perform the conversion
    transform_markdown(
        pdf_path=pdf_path,
        markdown_path=markdown_path,
        markdown_assets_path=Path("images"),
        analysing_path=analysing_path if generate_plot else None,
        models_cache_path=models_cache_path,
        includes_footnotes=includes_footnotes,
        generate_plot=generate_plot,
        on_ocr_event=on_ocr_event,
    )

    return {
        "markdown_path": str(markdown_path),
        "assets_dir": str(assets_dir)
    }
