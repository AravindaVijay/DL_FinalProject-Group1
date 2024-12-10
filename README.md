# DL_FinalProject-Group1
This is the repository of the Final_Project for Group 1, Fall 2024 GWU Data Science DL


# VisionVoice: Accessible Image Captioning with Audio

## Project Overview
VisionVoice is a deep learning-powered application designed to generate textual descriptions for images and convert these captions into audio for accessibility. This project leverages state-of-the-art computer vision and natural language processing techniques to bridge the gap between technology and accessibility, making visual content comprehensible to visually impaired users.

---

## Features
- **Image Captioning**: Uses advanced models like ResNet50 + LSTM, EfficientNet-B3 + LSTM, and BLIP for descriptive captions.
- **Text-to-Speech (TTS)**: Converts generated captions into natural-sounding audio.
- **Custom Models**: Includes CNN + RNN models built from scratch.
- **Pre-Trained Models**: Incorporates the BLIP base model for enhanced captioning performance.
- **Beam Search**: Employs advanced decoding for improved accuracy.

---

## Getting Started

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/AravindaVijay/DL_FinalProject-Group1.git
   cd DL_FinalProject-Group1
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the [COCO 2017 dataset](https://cocodataset.org/#download) and place the following:
   - **Training Images**: `Data/train2017/`
   - **Validation Images**: `Data/val2017/`
   - **Annotations**: `Data/annotations/`

---

## Dataset
### COCO 2017 Dataset
- **Images**: Each image has 5–7 human-generated captions.
- **Split**:
  - Training: ~118,000 images
  - Validation: ~5,000 images
- **Preprocessing**:
  - Resizing to 224x224
  - Normalization
  - Caption tokenization using the provided vocabulary script.

---

## Architecture and Methodology

### Model Architectures
1. **Custom CNN + RNN**:
   - **Encoder**: Custom CNN with 3 Conv2D layers and Adaptive Average Pooling.
   - **Decoder**: LSTM-based RNN for sequential caption generation.
   - **Functionality**: Combines convolutional layers for visual feature extraction with recurrent layers for language generation.

2. **EfficientNet-B3 + LSTM**:
   - **Encoder**: Pre-trained EfficientNet-B3 for feature extraction.
   - **Decoder**: LSTM with Beam Search for advanced captioning.

3. **ResNet50 + LSTM**:
   - **Encoder**: Pre-trained ResNet50 with a custom fully connected layer.
   - **Decoder**: LSTM for generating sequential captions.

4. **BLIP (Pre-Trained)**:
   - **Encoder**: Vision Transformer (ViT).
   - **Decoder**: Transformer-based language model.

### Training and Evaluation
- **Loss Function**: Cross-Entropy Loss
- **Optimization**: Adam Optimizer with learning rate scheduling.
- **Evaluation Metrics**:
  - BLEU Score for caption accuracy.
  - Qualitative analysis of generated captions.

---

## File Structure
```
DL_FinalProject-Group1/
├── Code/
│   ├── app.py                 # Main application script
│   ├── data_loader.py         # Data preprocessing and loading
│   ├── vocabulary.py          # Vocabulary handling
│   ├── models/
│   │   ├── resnet_lstm.py     # ResNet50 + LSTM model
│   │   ├── efficientnet_lstm.py # EfficientNet-B3 + LSTM model
│   │   ├── custom_cnn_rnn.py  # Custom CNN + RNN model
│   │   ├── blip_model.py      # BLIP pre-trained model
├── Data/
│   ├── train2017/             # Training images
│   ├── val2017/               # Validation images
│   ├── annotations/           # Annotations JSON files
├── requirements.txt           # Required Python libraries
├── checkpoints/               # Saved model weights
└── README.md                  # Project documentation
```

---

## Running the Project

### 1. Training
Train a model using its respective script. For example:
```bash
python Code/models/resnet_lstm.py
```

### 2. Generating Captions
Generate captions for new images:
```bash
python Code/app.py --image_path <path_to_image>
```

### 3. Testing
Evaluate the model on validation data:
```bash
python Code/app.py --mode test
```

---

## Results
### BLEU Scores:
- **ResNet50 + LSTM**: 0.1487
- **EfficientNet-B3 + LSTM**: 0.2002
- **Custom CNN + RNN**: 
- **BLIP (Pre-trained)**:

### Observations:
- BLIP performs best for detailed captions.
- EfficientNet achieves a balance of speed and performance.

---

## Future Work
- Explore attention mechanisms for enhanced captioning.
- Improve TTS module for more natural audio output.
- Train with larger datasets for improved accuracy.

---

## Acknowledgments
- [COCO Dataset](https://cocodataset.org/) for benchmark images and captions.
- [PyTorch](https://pytorch.org/) for deep learning frameworks.
- [BLIP Model](https://huggingface.co/Salesforce/blip) for pre-trained captioning.

---
