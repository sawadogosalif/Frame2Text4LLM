

import numpy as np
import logging

logger = logging.getLogger(__name__)

class PaddleOCR:
    """
    PaddleOCR implementation.
    """
    
    def __init__(self):
        """Initialize PaddleOCR."""
        self._check_paddleocr()
    
    def _check_paddleocr(self):
        """Check if PaddleOCR is installed."""
        try:
            from paddleocr import PaddleOCR as OCREngine
            logger.info("PaddleOCR is available")
        except ImportError:
            logger.warning("PaddleOCR not installed. Install with: pip install paddleocr")
    
    def process_image(self, image: np.ndarray, lang: str = "en") -> str:
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image: NumPy array containing the image
            lang: Language code (for PaddleOCR, use 'en', 'ch', etc.)
            
        Returns:
            Extracted text from the image
        """
        try:
            from paddleocr import PaddleOCR
            
            # Map common language codes to PaddleOCR language codes
            paddle_lang = lang
            if lang == "eng":
                paddle_lang = "en"
            elif lang == "fra":
                paddle_lang = "fr"
            elif lang == "deu":
                paddle_lang = "german"
            elif lang == "jpn":
                paddle_lang = "japan"
            elif lang == "kor":
                paddle_lang = "korean"
                        
            # raisoe if error is code is not  coorect
            ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
            result = ocr.ocr(image, cls=True)
            
            text_results = []
            for line in result:
                for word_info in line:
                    text = word_info[1][0]
                    score = word_info[1][1]
                    if score > 0.5:
                        text_results.append(text)
            
            return " ".join(text_results)
        except Exception as e:
            raise RuntimeError(f"PaddleOCR failed: {str(e)}")