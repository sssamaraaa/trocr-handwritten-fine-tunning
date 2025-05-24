import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description="Train a TrOCR model for handwritten text recognition")

    parser.add_argument("--data", type=str, required=True, help="Path to root data")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--cont", type=str, required=False, default=None,
                        help="Path to existing version folder (e.g., versions/v3) to continue training")

    return parser.parse_args()

def parse_infer_args():
    parser = argparse.ArgumentParser(description="Checking the operation of the model")

    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--device", type=str, default="cpu", help="Выбери устройство: 'cpu' или номер GPU, например '0'")

    return parser.parse_args()