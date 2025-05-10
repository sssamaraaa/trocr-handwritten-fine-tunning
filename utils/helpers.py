import os

def create_signal_handler(model, processor, epoch_completed, last_save_path):
    import sys, logging
    def handler(sig, frame):
        logging.info('\nCtrl+C пойман.')
        if epoch_completed > 0:
            logging.info('Сохраняем модель...')
            model.save_pretrained(os.path.join(last_save_path, "model"))
            processor.save_pretrained(os.path.join(last_save_path, "processor"))
        sys.exit(0)

    return handler

def create_plot_metrics(root, train_losses, val_losses, cer_scores, wer_scores):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Eval Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cer_scores) + 1), cer_scores, label="CER")
    plt.plot(range(1, len(wer_scores) + 1), wer_scores, label="WER")
    plt.title("Error Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    graphs_dir = os.path.join(root, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    plt.savefig(os.path.join(graphs_dir, "training_metrics.png"))
    plt.close()

def get_next_save_path(versions_dir):
    versions_num_list = [
        int(v_num[1:]) for v_num in os.listdir(versions_dir)
        if v_num.startswith('v') and v_num[1:].isdigit()
    ]
    max_num = max(versions_num_list) if versions_num_list else 0

    return f"v{max_num + 1}"

