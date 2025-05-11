import os
import torch
import signal
import logging
from utils.data.dataset import HandwrittenTextDataset
from utils.training.early_stopping import EarlyStopping
from utils.argparser import parse_args
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


if __name__ == "__main__":
    set_seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = parse_args()
    data_root = args.data
    root = os.path.dirname(os.path.abspath(__file__))
    version_dir = get_next_save_path(os.path.join(root, "versions"))
    last_save_path = os.path.join(version_dir, "last")
    checkpoint_path = os.path.join(version_dir, "checkpoint")
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(last_save_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    train_dir = os.path.join(data_root, "train", "images_train")
    train_labels = os.path.join(data_root, "train", "train.tsv")
    val_dir = os.path.join(data_root, "test", "images_test")
    val_labels = os.path.join(data_root, "test", "test.tsv")

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
        logging.info("Loading model from a checkpoint...")
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

        tqdm.write(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {avg_val_loss:.4f}")
        tqdm.write(f"CER: {cer_score:.4f}, WER: {wer_score:.4f}")

        scheduler.step(avg_val_loss)

        save_checkpoint(last_save_path, model, processor, optimizer, scheduler, scaler, epoch_completed, train_losses,
                        val_losses, cer_scores, wer_scores)

        early_stopping(avg_val_loss, model, processor)
        if early_stopping.early_stop:
            logging.info("Early stopping!")
            break

    model.save_pretrained(os.path.join(last_save_path, "model"))
    processor.save_pretrained(os.path.join(last_save_path, "processor"))
    logging.info("The training has been successfully completed!")

    create_plot_metrics(root, train_losses, val_losses, cer_scores, wer_scores)