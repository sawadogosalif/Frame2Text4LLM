import os
import numpy as np
from loguru import logger
import shutil
import pytesseract
import platform

class TesseractOCR:
    """
    Tesseract OCR implementation.
    """
    
    def __init__(self):
        """Initialize Tesseract OCR."""
        self._configure_tesseract()
        self.available_languages = self.get_available_languages()
    
    def _configure_tesseract(self):
        """Configure Tesseract path based on OS."""

        logger.info("🔍 Trying to find tesseract in system PATH")
        tesseract_path = shutil.which("tesseract")
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"✅ Tesseract found: {tesseract_path}")
            return tesseract_path

        system = platform.system()

        if system == "Windows":
            default_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(path), "tessdata")
                    logger.info(f"✅ Tesseract found at fallback Windows path: {path}")
                    return path

        raise EnvironmentError("❌ Tesseract executable not found. Please install Tesseract and make sure it's in your PATH.")

    
    def get_available_languages(self) -> list:
        """
        List available languages in the Tesseract installation.
        """
        try:
            langs = pytesseract.get_languages(config='')
            logger.info(f"📚 Available Tesseract languages: {langs}")
            return langs
        except Exception as e:
            logger.warning(f"⚠️ Failed to list languages: {e}")
            return []

    
    def process_image(self, image: np.ndarray, lang: str = "eng") -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image: NumPy array containing the image
            lang: Language code ('eng', 'fra', etc.)
            
        Returns:
            Extracted text from the image
        """
        try:
            if lang not in self.available_languages:
                logger.warning(f"⚠️ Language '{lang}' not available. Falling back to 'eng'.")
                logger.info(f"📚 Available Tesseract languages: {self.available_languages}")
                lang = "eng"

            config = '--psm 6'  # BON on garde ça pour le momnt
            
            text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
            logger.debug(f"📝 OCR Result (lang={lang}): {text[:100]}...")
            return text
            
        except Exception as e:
            logger.error(f"❌ Tesseract OCR failed: {str(e)}")
            raise RuntimeError(f"Tesseract OCR failed: {str(e)}")
