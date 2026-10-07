# AI/ML Fundamentals: Image and Text Classification

Two small, end-to-end deep learning projects in PyTorch.

**Stack:** Python · PyTorch · Neural Networks · CNN · NLP · NumPy · Pandas · Matplotlib · scikit-learn · Docker

## 1. CNN image classifier (`cnn_image_classifier.py`)
- Classifies Fashion-MNIST clothing images into 10 classes.
- Two Conv -> ReLU -> MaxPool blocks, then a fully connected classifier with Dropout.
- NumPy for per-class accuracy; Matplotlib for training curves and sample predictions.
- Test accuracy: **91.0%** after 5 epochs (CPU training)

## 2. NLP text classifier (`nlp_text_classifier.py`)
- Classifies news posts into 4 topics (space, baseball, graphics, politics).
- Pandas for loading and exploring the data (class counts, word-count statistics).
- TF-IDF features feed a feed-forward neural network (PyTorch).
- Test accuracy: **90.5%** after 10 epochs (precision, recall and F1 about 0.88-0.93 per topic)

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
- On the text task, test accuracy levelled off at about 90.5% from epoch 5 while the training loss kept falling (1.32 to 0.07). That is mild overfitting, so more epochs would not help; dropout, more data or a pretrained model would.
- Baseball was the easiest topic (F1 0.93); space and politics were slightly harder (F1 0.88 and 0.89).
- TF-IDF features plus a small neural network already work well for topic classification, and the vectorizer must be fitted on the training split only to avoid data leakage.
- <!-- ADD ONE LINE in your own words about the CNN, after looking at outputs/cnn/sample_predictions.png: which clothing items it got wrong -->

## Possible improvements
Data augmentation and a deeper CNN (or a pretrained ResNet) for images; transformer embeddings such as BERT for text.
