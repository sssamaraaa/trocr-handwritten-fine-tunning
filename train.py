import os
import torch
import signal
import logging
from utils.data.dataset import HandwrittenTextDataset
from utils.training.early_stopping import EarlyStopping
from utils.argparser import parse_train_args
from utils.helpers import create_signal_handler, create_plot_metrics, get_next_save_path
from utils.training.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import set_seed
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    set_seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_train_args()
    data_root = args.data
    root = os.path.dirname(os.path.abspath(__file__))

    if args.cont:
        load_version = args.cont
        load_path_last = os.path.join(load_version, "last")

        save_version_dir = get_next_save_path(os.path.join(root, "versions"))
        save_path_last = os.path.join(save_version_dir, "last")
        os.makedirs(save_path_last, exist_ok=True)
        logging.info(f"Continuing training from {load_version}, saving to new version: {save_version_dir}")
    else:
        save_version_dir = get_next_save_path(os.path.join(root, "versions"))
        save_path_last = os.path.join(save_version_dir, "last")
        os.makedirs(save_path_last, exist_ok=True)
        logging.info(f"Starting new training run at: {save_version_dir}")

    train_csv = os.path.join(data_root, "train.csv")
    images_path = os.path.join(data_root, "images")

    df = pd.read_csv(train_csv, header=0, encoding="utf-8-sig")
    df = df[["text", "name"]]
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tmp_train_csv = os.path.join(save_version_dir, "train_split.csv")
    tmp_val_csv = os.path.join(save_version_dir, "val_split.csv")
    train_df.to_csv(tmp_train_csv, index=False, header=False)
    val_df.to_csv(tmp_val_csv, index=False, header=False)

    if args.cont:
        load_version = args.cont
        load_path_last = os.path.join(load_version, "last")
        processor = TrOCRProcessor.from_pretrained(os.path.join(load_path_last, "processor"))
        model = VisionEncoderDecoderModel.from_pretrained(os.path.join(load_path_last, "model"))
    else:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = HandwrittenTextDataset(images_path, tmp_train_csv, processor, augment=True)
    val_ds = HandwrittenTextDataset(images_path, tmp_val_csv, processor, augment=False)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True, min_lr=1e-8)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=3, save_path=os.path.join(save_version_dir, "best_model"))

    metric_cer = evaluate.load("cer")
    metric_wer = evaluate.load("wer")

    if args.cont:
        epoch_completed, train_losses, val_losses, cer_scores, wer_scores = load_checkpoint(
            load_path_last, model, optimizer, scheduler, scaler
        )
    else:
        epoch_completed = 0
        train_losses, val_losses, cer_scores, wer_scores = [], [], [], []

    signal.signal(signal.SIGINT, create_signal_handler(model, processor, epoch_completed, save_path_last))

    for epoch in range(epoch_completed, args.epochs):
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

        tqdm.write(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {avg_val_loss:.4f}")
        tqdm.write(f"CER: {cer_score:.4f}, WER: {wer_score:.4f}")

        scheduler.step(avg_val_loss)

        save_checkpoint(
            save_path_last,
            model, processor, optimizer, scheduler, scaler,
            epoch + 1, train_losses, val_losses, cer_scores, wer_scores
        )

        early_stopping(avg_val_loss, model, processor)
        if early_stopping.early_stop:
            logging.info("Early stopping!")
            break

    model.save_pretrained(os.path.join(save_path_last, "model"))
    processor.save_pretrained(os.path.join(save_path_last, "processor"))
    logging.info("The training has been successfully completed!")

    create_plot_metrics(root, train_losses, val_losses, cer_scores, wer_scores, save_version_dir)

