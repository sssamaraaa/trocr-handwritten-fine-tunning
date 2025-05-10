import os
import pandas as pd
import torch
import signal
import logging
from utils.early_stopping import EarlyStopping
from utils.argparser import parse_args
from utils.helpers import create_signal_handler, create_plot_metrics, get_next_save_path
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed
from PIL import Image, UnidentifiedImageError, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import functional as TF
import random
import evaluate

set_seed(42)

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
        image = Image.open(self.images[idx]).convert("RGB").resize((384, 384))
        if self.augment:
            image = self.augment_image(image)
        encoding = self.processor(images=image, text=self.texts[idx], return_tensors="pt", padding="max_length",
                                  truncation=True, max_length=self.max_length)
        return {
            "pixel_values": encoding.pixel_values.squeeze(),
            "labels": encoding.labels.squeeze(),
            "text": self.texts[idx]
        }


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    version_dir = get_next_save_path(os.path.join(root, "versions"))
    last_save_path = os.path.join(version_dir, "last")
    checkpoint_path = os.path.join(version_dir, "checkpoint")
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(last_save_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    train_dir = os.path.join(data_root, "train", "images_train")
    train_labels = os.path.join(data_root, "dataset_rus", "train", "train.tsv")
    val_dir = os.path.join(data_root, "dataset_rus", "test", "images_test")
    val_labels = os.path.join(data_root, "dataset_rus", "test", "test.tsv")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = HandwrittenTextDataset(train_dir, train_labels, processor, augment=True)
    val_ds = HandwrittenTextDataset(val_dir, val_labels, processor, augment=False)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True, min_lr=1e-6)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=3, save_path=os.path.join(version_dir, "best_model"))

    metric_cer = evaluate.load("cer")
    metric_wer = evaluate.load("wer")

    if args.cont:
        logging.info("Загрузка модели из чекпоинта...")
        load_checkpoint(checkpoint_path, model, processor, optimizer, scheduler, scaler)

    epoch_completed = 0
    signal.signal(signal.SIGINT, create_signal_handler(model, processor, epoch_completed, last_save_path))
    train_losses, val_losses = [], []
    cer_scores, wer_scores = [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        loop = tqdm(train_dl, desc=f"[Epoch {epoch + 1}] Training")

        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_completed += 1
        avg_train_loss = running_loss / len(train_dl)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        predictions, references = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validating"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                texts = batch["text"]

                with autocast():
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()

                    generated_ids = model.generate(pixel_values)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    predictions.extend(generated_texts)
                    references.extend(texts)

        avg_val_loss = val_loss / len(val_dl)
        val_losses.append(avg_val_loss)

        cer_score = metric_cer.compute(predictions=predictions, references=references)
        wer_score = metric_wer.compute(predictions=predictions, references=references)
        cer_scores.append(cer_score)
        wer_scores.append(wer_score)

        logging.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {avg_val_loss:.4f}")
        logging.info(f"CER: {cer_score:.4f}, WER: {wer_score:.4f}")

        scheduler.step(avg_val_loss)

        save_checkpoint(last_save_path, model, processor, optimizer, scheduler, scaler, epoch_completed, train_losses,
                        val_losses, cer_scores, wer_scores)

        early_stopping(avg_val_loss, model, processor)
        if early_stopping.early_stop:
            logging.info("Early stopping!")
            break

    model.save_pretrained(os.path.join(last_save_path, "model"))
    processor.save_pretrained(os.path.join(last_save_path, "processor"))
    logging.info("Обучение завершено.")

    create_plot_metrics(root, train_losses, val_losses, cer_scores, wer_scores)