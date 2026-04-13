"""
OpenAI GPT-4V / GPT-4o 视觉解析后端
需要设置环境变量 OPENAI_API_KEY
"""
import json
import os
import base64
import io
from .base import PerceptionBackend
from ..schemas.models import UIElement


class OpenAIVisionBackend(PerceptionBackend):
    """
    基于 OpenAI Vision API 的截图理解后端。
    适用于没有本地 GPU 但持有 API Key 的场景。
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "Missing openai SDK. Install: pip install openai"
            ) from e

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _encode_image(self, image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def parse(self, image) -> tuple[list[UIElement], str]:
        b64_image = self._encode_image(image)
        prompt = (
            "Analyze this screenshot. List all interactive UI elements "
            "(buttons, inputs, links, checkboxes) as a JSON array with fields: "
            "element_type, text, bbox [x1,y1,x2,y2]."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        output_text = response.choices[0].message.content or ""
        elements = self._extract_json_elements(output_text)
        return elements, output_text

    def _extract_json_elements(self, text: str) -> list[UIElement]:
        elements = []
        try:
            import re
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
