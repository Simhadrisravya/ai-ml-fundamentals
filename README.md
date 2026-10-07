# AI/ML Fundamentals: Image and Text Classification

Two small, end-to-end deep learning projects in PyTorch.

**Stack:** Python · PyTorch · Neural Networks · CNN · NLP · NumPy · Pandas · Matplotlib · scikit-learn · Docker

## 1. CNN image classifier (`cnn_image_classifier.py`)
- Classifies Fashion-MNIST clothing images into 10 classes.
- Two Conv -> ReLU -> MaxPool blocks, then a fully connected classifier with Dropout.
- NumPy for per-class accuracy; Matplotlib for training curves and sample predictions.
- Test accuracy: **XX.X%** <!-- replace with your real result -->

## 2. NLP text classifier (`nlp_text_classifier.py`)
- Classifies news posts into 4 topics (space, baseball, graphics, politics).
- Pandas for loading and exploring the data (class counts, word-count statistics).
- TF-IDF features feed a feed-forward neural network (PyTorch).
- Test accuracy: **XX.X%** <!-- replace with your real result -->

## Run locally
```bash
pip install -r requirements.txt
python cnn_image_classifier.py
python nlp_text_classifier.py
```
Results (plots) are saved in `outputs/`.

## Run with Docker
```bash
docker build -t ai-ml-fundamentals .
docker run --rm -v "$(pwd)/outputs:/app/outputs" ai-ml-fundamentals
```

## What I learned
<!-- 2-3 lines in your own words: what the curves show, which classes were confused -->
