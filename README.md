# TrOCR Handwritten Fine-Tuning
This project is an implementation of training with additional tuning
of Microsoft's TrOCR model for handwriting recognition.
It includes a full cycle: from data preparation to training,
checkpoint saving, early stop and inference.

## Project structure
```
trocr-handwritten-fine-tunning/
├── train.py               
├── inference.py           
├── requirements.txt       
├── utils/
│   ├── argparser.py       # Command line argument parsing
│   ├── image/
│   │   └── resize.py      # Resize image for inference
│   ├── data/
│   │   └── dataset.py     # Dataset class
│   ├── training/
│   │   ├── checkpoint.py       # Save/load checkpoint
│   │   └── early_stopping.py   # Earlystopping class
│   ├── helpers.py         # Helper function
│   ├── argparser.py       # Command line argument parsing
│   └── seed.py            # Set seed
├── versions/              # Saved versions of models
└── metrics/               # Saved metrics
```
## Dataset format
The dataset directory should contain
a folder with images and a csv file with annotations.
```
dataset/
├──images
└──train.csv
```
The annotations CSV file must contain two columns: `text` and `name`, where:

- `text` is the label (e.g. handwritten content)
- `name` is the corresponding image filename

## Metrics
During training, the following metrics are calculated:
* CER (Character Error Rate) is an indicator of errors at the character level.
* WER (Word Error Rate) is an indicator of errors at the word level.

The results are saved and visualized as graphs in the metrics folder.

## Train and inference
Training and inference are run via the command line:
```
python train.py --dataset (path to dataset folder)
                --epochs(number) 
                --cont (path to version for continue training)
                
python inference.py --weights (path to folder with model/tokenizer) 
                    --image (path to image)
                    --device (0-cuda, 1-cpu)
```
If you specify an argument for --cont, the training results will be saved to a new folder,
and the previous folder will remain intact.

### License
This project is licensed under the MIT License.
