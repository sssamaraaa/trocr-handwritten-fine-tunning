import os
import logging
from PIL import Image
from utils.argparser import parse_infer_args
from utils.image.resize import resize_with_aspect_and_padding
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


if __name__ == "__main__":
    args = parse_infer_args()
    weights = args.weights
    device = args.device

    if args.device.isdigit():
        device = f"cuda:{args.device}"
    elif args.device.lower() == "cpu":
        device = "cpu"
    else:
        raise ValueError("Неверное значение для --device. Используй 'cpu' или номер GPU, например '0'")

    model = VisionEncoderDecoderModel.from_pretrained(os.path.join(weights, "model")).to(device)
    processor = TrOCRProcessor.from_pretrained(os.path.join(weights, "processor"))

    image_path = args.image
    recognized_text = ""
    try:
        # image = resize_with_aspect_and_padding(Image.open(image_path).convert("RGB"))
        image = Image.open(image_path).convert("RGB").resize((384, 384))
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_new_tokens=30)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_text += text + ""
    except Exception as e:
        logging.info(f"Ошибка при обработке изображения: {e}")

    print(recognized_text)

