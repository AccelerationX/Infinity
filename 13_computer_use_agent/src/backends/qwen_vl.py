"""
Qwen2-VL 视觉解析后端
在本地加载 Qwen/Qwen2-VL-2B-Instruct 进行 UI 元素识别
"""
import json
import re
from .base import PerceptionBackend
from ..schemas.models import UIElement


class QwenVLBackend(PerceptionBackend):
    """
    基于 Qwen2-VL 的截图理解后端。
    首次加载时会自动从 HuggingFace 下载模型（约 4GB）。
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise RuntimeError(
                "Missing dependencies for QwenVLBackend. "
                "Install: pip install torch transformers qwen-vl-utils accelerate"
            ) from e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._process_vision_info = process_vision_info

    def parse(self, image) -> tuple[list[UIElement], str]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Analyze this screenshot. List all interactive UI elements "
                            "(buttons, inputs, links, checkboxes) as a JSON array with fields: "
                            "element_type, text, bbox [x1,y1,x2,y2]."
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elements = self._extract_json_elements(output_text)
        return elements, output_text

    def _extract_json_elements(self, text: str) -> list[UIElement]:
        elements = []
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                for i, item in enumerate(data):
                    elements.append(
                        UIElement(
                            element_id=f"elem_{i}",
                            element_type=item.get("element_type", "unknown"),
                            text=item.get("text", ""),
                            bbox=item.get("bbox", []),
                            confidence=item.get("confidence", 0.0),
                        )
                    )
        except Exception:
            pass
        return elements
