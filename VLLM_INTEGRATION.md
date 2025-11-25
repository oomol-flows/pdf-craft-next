# vLLM Integration for PDF-to-Markdown

## Overview

This project now integrates **vLLM** (Very Large Language Model inference) for high-performance PDF OCR processing using DeepSeek-OCR. vLLM provides significantly faster inference compared to standard transformers, especially beneficial for processing large PDFs.

## What Changed

### 1. Modified Libraries (in `/app/workspace/libs/`)

We copied and modified the open-source `pdf_craft` and `doc_page_extractor` libraries to support vLLM:

#### **libs/doc_page_extractor/**
- **model_vllm.py** (NEW): vLLM-based DeepSeek-OCR model implementation
  - Uses `vllm.LLM` for high-performance inference
  - Supports configurable GPU memory utilization
  - Optimized for batch processing

- **extractor.py** (MODIFIED): Added `backend` parameter
  - `backend="transformers"`: Original implementation
  - `backend="vllm"`: New vLLM-based implementation (default in our task)
  - Additional parameters: `gpu_memory_utilization`, `max_model_len`

#### **libs/pdf_craft/**
- **pdf/extractor.py** (MODIFIED): Pass backend parameters through Extractor class
- **pdf/ocr.py** (MODIFIED): Pass backend parameters to ocr_pdf function
- **transform.py** (MODIFIED): Added backend parameters to transform_markdown function

### 2. Updated Task

**tasks/pdf-to-markdown/__init__.py**:
- Added `sys.path` manipulation to use local modified libraries
- Enabled vLLM backend by default with optimized settings:
  - `backend="vllm"`
  - `gpu_memory_utilization=0.9` (use 90% of GPU memory)
  - `max_model_len=4096` (maximum sequence length)

## Architecture

```
┌─────────────────────────────────────┐
│  pdf-to-markdown task               │
│  (uses local libs via sys.path)     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  libs/pdf_craft/transform.py        │
│  + backend parameter                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  libs/pdf_craft/pdf/ocr.py          │
│  + backend parameter                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  libs/pdf_craft/pdf/extractor.py    │
│  + backend parameter                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  libs/doc_page_extractor/           │
│  extractor.py                       │
│  + backend selection logic          │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐ ┌──────────────────┐
│   model.py   │ │  model_vllm.py   │
│ transformers │ │     vLLM         │
└──────────────┘ └──────────────────┘
```

## Performance Benefits

vLLM provides several advantages over standard transformers:

1. **PagedAttention**: Efficient memory management for KV caches
2. **Continuous Batching**: Better GPU utilization
3. **Optimized CUDA Kernels**: Faster inference
4. **CUDA Graphs**: Reduced kernel launch overhead
5. **Automatic Quantization Support**: Optional FP8/INT8 quantization

## Usage

The task automatically uses vLLM. No changes needed for basic usage.

### Testing

Run the existing flow to test vLLM integration:

```bash
# The task will automatically use vLLM backend
# Watch for log messages:
# [PageExtractor] Using vLLM backend for high-performance inference
# [vLLM] Loading DeepSeek-OCR with vLLM backend...
```

### Switching Back to Transformers (if needed)

To use the original transformers backend, modify the task file:

```python
transform_markdown(
    ...
    backend="transformers",  # Change from "vllm"
)
```

## Configuration

### vLLM Parameters

Adjustable in `tasks/pdf-to-markdown/__init__.py`:

```python
transform_markdown(
    ...
    backend="vllm",
    gpu_memory_utilization=0.9,  # 0.0-1.0, how much GPU memory to use
    max_model_len=4096,           # Maximum sequence length
)
```

### GPU Memory Utilization

- `0.9` (default): Use 90% of GPU memory, leaving 10% for other processes
- Adjust based on your GPU memory and concurrent workloads
- Higher values = better performance but less memory for other tasks

### Max Model Length

- `4096` (default): Suitable for most OCR tasks
- Increase for very long documents or complex layouts
- Decrease to save memory if processing simple pages

## Known Limitations

1. **First Load**: vLLM model loading takes ~30 seconds on first use
2. **Memory**: Requires sufficient GPU memory (recommended: ≥16GB)
3. **Cold Start**: First inference may be slower due to CUDA graph compilation

## Troubleshooting

### Out of Memory Errors

Reduce `gpu_memory_utilization`:
```python
gpu_memory_utilization=0.7  # Use 70% instead of 90%
```

### Slow First Page

This is normal - vLLM compiles CUDA graphs on first inference. Subsequent pages will be much faster.

### Import Errors

Ensure the modified libraries are in `/app/workspace/libs/` and the task correctly adds them to `sys.path`.

## Technical Details

### DeepSeek-OCR with vLLM

vLLM supports DeepSeek-OCR natively through:
- `vllm.model_executor.models.deepseek_ocr`
- Multimodal processing with image inputs
- Optimized attention mechanisms for vision-language models

### Image Processing

Images are passed to vLLM using:
```python
{
    "prompt": "<image>\n{prompt}",
    "multi_modal_data": {"image": image_path}
}
```

vLLM handles image encoding and processing automatically.

## Future Improvements

1. Add batch processing support (process multiple pages simultaneously)
2. Add quantization options (FP8, INT8) for memory efficiency
3. Expose more vLLM parameters (tensor parallel size, etc.)
4. Add performance benchmarking tools

## Credits

- **vLLM**: https://github.com/vllm-project/vllm
- **pdf-craft**: Open-source PDF processing library
- **doc-page-extractor**: DeepSeek-OCR integration library
- **DeepSeek-OCR**: deepseek-ai/DeepSeek-OCR

## Support

For issues specific to:
- **vLLM integration**: Check this document and vLLM logs
- **OCR quality**: Same as before (model-dependent)
- **PDF processing**: Same as before (pdf-craft-dependent)
