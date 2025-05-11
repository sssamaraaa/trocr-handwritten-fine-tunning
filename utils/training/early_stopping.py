class EarlyStopping:
    def __init__(self, patience=3, verbose=True, save_path="best_model"):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.save_path = save_path
        self.verbose = verbose

    def __call__(self, val_loss, model, processor):
        import logging, os
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            model.save_pretrained(os.path.join(self.save_path, "model"))
            processor.save_pretrained(os.path.join(self.save_path, "processor"))
            if self.verbose: logging.info(f"Loss has been improved\nSaving model")
        else:
            self.counter += 1
            if self.verbose: logging.info(f"No improvements: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True