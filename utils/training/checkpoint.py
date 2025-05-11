import torch
import os

def save_checkpoint(path, model, processor, optimizer, scheduler, scaler, epoch, train_losses, val_losses, cer_scores, wer_scores):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'cer_scores': cer_scores,
        'wer_scores': wer_scores,
    }

    torch.save(checkpoint, os.path.join(path, 'checkpoint.pth'))
    model.save_pretrained(os.path.join(path, "model"))
    processor.save_pretrained(os.path.join(path, "processor"))

def load_checkpoint(path, model, processor, optimizer, scheduler, scaler):
    checkpoint_path = os.path.join(path, 'checkpoint.pth')
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    processor.load_pretrained(os.path.join(path, "processor"))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return (
        checkpoint['epoch'],
        checkpoint['train_losses'],
        checkpoint['val_losses'],
        checkpoint['cer_scores'],
        checkpoint['wer_scores']
    )
