def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train a TrOCR model for handwritten text recognition.")
    parser.add_argument("--data", type=str, required=True, help="Path to root data")
    parser.add_argument("--cont", action="store_true", help="Continue training")
    parser.add_argument("--epochs", type=str, required=True, help="Number of epochs")

    return parser.parse_args()