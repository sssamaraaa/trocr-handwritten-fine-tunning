import random
import os
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError, ImageEnhance
from torch.utils.data import Dataset
from utils.image.resize import resize_with_aspect_and_padding
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import functional as TF


class HandwrittenTextDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor, max_length=50, augment=False):
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        df = pd.read_csv(annotation_file, sep='\t', header=None, names=['filename', 'text'])
        df['full_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))

        def is_valid(row):
            try:
                Image.open(row['full_path']).convert("RGB")
                return True
            except (FileNotFoundError, UnidentifiedImageError):
                return False

        with ThreadPoolExecutor() as executor:
            valid_mask = list(executor.map(is_valid, [row for _, row in df.iterrows()]))

        self.images = df['full_path'][valid_mask].tolist()
        self.texts = df['text'][valid_mask].tolist()

    def __len__(self):
        return len(self.images)

    def augment_image(self, image):
        if random.random() < 0.5:
            if random.random() < 0.5:
                angle = random.uniform(-7, 7)
                image = image.rotate(angle)

            if random.random() < 0.8:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

            image_tensor = TF.to_tensor(image)

            if random.random() < 0.7:
                angle = 0
                translations = [int(random.uniform(-0.1, 0.1) * image.width),
                                int(random.uniform(-0.1, 0.1) * image.height)]
                scale = random.uniform(0.9, 1.1)
                shear = [random.uniform(-7, 7), random.uniform(-7, 7)]
                image_tensor = TF.affine(image_tensor, angle=angle, translate=translations, scale=scale, shear=shear)

            if random.random() < 0.2:
                image_tensor = TF.gaussian_blur(image_tensor, kernel_size=[3, 3])

            if random.random() < 0.1:
                noise = torch.randn_like(image_tensor) * 0.02
                image_tensor = torch.clamp(image_tensor + noise, 0, 1)

            image = TF.to_pil_image(image_tensor)

        return image

    def __getitem__(self, idx):
        image = resize_with_aspect_and_padding(Image.open(self.images[idx]).convert("RGB"))
        if self.augment:
            image = self.augment_image(image)
        encoding = self.processor(images=image, text=self.texts[idx], return_tensors="pt", padding="max_length",
                                  truncation=True, max_length=self.max_length)
        return {
            "pixel_values": encoding.pixel_values.squeeze(),
            "labels": encoding.labels.squeeze(),
            "text": self.texts[idx]
        }