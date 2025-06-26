
from .tesseract import TesseractOCR
from .paddleocr import PaddleOCR
from .openaiocr import OpenAIOCR
from .mistralocr import MistralOCR

OCR_TOOLS = {
    "tesseract": TesseractOCR,
    "paddleocr": PaddleOCR,
    "openai": OpenAIOCR,
    "mistral": MistralOCR,
}
