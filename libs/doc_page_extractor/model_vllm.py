import os
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import encode_image_base64

DeepSeekOCRSize = Literal["tiny", "small", "base", "large", "gundam"]

@dataclass
class _SizeConfig:
    base_size: int
    image_size: int
    crop_mode: bool

_SIZE_CONFIGS: dict[DeepSeekOCRSize, _SizeConfig] = {
    "tiny": _SizeConfig(base_size=512, image_size=512, crop_mode=False),
    "small": _SizeConfig(base_size=640, image_size=640, crop_mode=False),
    "base": _SizeConfig(base_size=1024, image_size=1024, crop_mode=False),
    "large": _SizeConfig(base_size=1280, image_size=1280, crop_mode=False),
    "gundam": _SizeConfig(base_size=1024, image_size=640, crop_mode=True),
}


class DeepSeekOCRModelVLLM:
    """
    vLLM-based DeepSeek-OCR model for high-performance inference.

    This implementation uses vLLM for significantly faster inference compared
    to the standard transformers implementation, especially for batch processing.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        dtype: str = "bfloat16"
    ) -> None:
        self._model_name = "deepseek-ai/DeepSeek-OCR"
        self._cache_dir = str(model_path) if model_path else None
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._llm: LLM | None = None

    def download(self) -> None:
        """Download model weights (vLLM handles this automatically on first load)"""
        # vLLM will download on first use, but we can trigger it explicitly
        self._ensure_model()

    def load(self) -> None:
        """Load the vLLM model into memory"""
        self._ensure_model()

    def _ensure_model(self) -> LLM:
        """Ensure the vLLM model is loaded"""
        if self._llm is None:
            print(f"[vLLM] Loading DeepSeek-OCR with vLLM backend...")
            print(f"[vLLM] GPU memory utilization: {self._gpu_memory_utilization}")
            print(f"[vLLM] Max model length: {self._max_model_len}")
            print(f"[vLLM] Data type: {self._dtype}")

            self._llm = LLM(
                model=self._model_name,
                trust_remote_code=True,
                dtype=self._dtype,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                download_dir=self._cache_dir,
                enforce_eager=False,  # Enable CUDA graphs for better performance
            )
            print(f"[vLLM] Model loaded successfully!")

        return self._llm

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temp_path: str,
        size: DeepSeekOCRSize
    ) -> str:
        """
        Generate OCR output for the given image using vLLM.

        Args:
            image: PIL Image to process
            prompt: Text prompt for OCR
            temp_path: Temporary directory for intermediate files
            size: Size configuration for the model

        Returns:
            Extracted text from the image
        """
        llm = self._ensure_model()
        config = _SIZE_CONFIGS[size]

        # Save image temporarily for reference
        temp_image_path = os.path.join(temp_path, "temp_image.png")
        image.save(temp_image_path)

        # Use the prompt as-is (it already contains <image> token from extractor.py)
        # The prompt format from extractor: "<image>\n<|grounding|>Convert the document to markdown."
        full_prompt = prompt

        # Configure sampling parameters for deterministic OCR
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic output for OCR
            max_tokens=self._max_model_len,
            stop=None,
        )

        # vLLM multimodal input format
        # IMPORTANT: vLLM expects PIL Image object, not path string
        outputs = llm.generate(
            prompts=[{
                "prompt": full_prompt,
                "multi_modal_data": {
                    "image": image  # Pass PIL Image directly, not path
                }
            }],
            sampling_params=sampling_params
        )

        # Extract the generated text
        text_result = outputs[0].outputs[0].text.strip()

        return text_result
