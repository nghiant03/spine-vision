"""Text recognition using VietOCR."""

import numpy as np
from loguru import logger
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


class TextRecognizer:
    """Wrapper for VietOCR text recognition.

    Recognizes Vietnamese text from cropped text region images.
    """

    def __init__(
        self,
        model_name: str = "vgg_transformer",
        device: str = "cuda:0",
        use_beamsearch: bool = False,
    ) -> None:
        """Initialize text recognizer.

        Args:
            model_name: VietOCR model configuration name.
            device: Device to run inference on ('cuda:0' or 'cpu').
            use_beamsearch: Whether to use beam search decoding.
        """
        logger.info(f"Loading recognition model: {model_name}")

        config = Cfg.load_config_from_name(model_name)
        config["device"] = device
        config["cnn"]["pretrained"] = False
        config["predictor"]["beamsearch"] = use_beamsearch

        self.model = Predictor(config)
        self.device = device

    def recognize(self, image: Image.Image) -> str:
        """Recognize text from a cropped image.

        Args:
            image: PIL Image of cropped text region.

        Returns:
            Recognized text string.
        """
        result = self.model.predict(image)
        return result if isinstance(result, str) else result[0]

    def recognize_from_array(self, image: np.ndarray) -> str:
        """Recognize text from a numpy array.

        Args:
            image: Image as numpy array (RGB).

        Returns:
            Recognized text string.
        """
        pil_image = Image.fromarray(image)
        return self.recognize(pil_image)

    def recognize_batch(self, images: list[Image.Image]) -> list[str]:
        """Recognize text from multiple images.

        Args:
            images: List of PIL Images.

        Returns:
            List of recognized text strings.
        """
        return [self.recognize(img) for img in images]
