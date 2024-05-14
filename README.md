# Arabic-font-recognition-system
In this project, we implement an Arabic Font Recognition System. Given an image containing a paragraph written in Arabic, the system is supposed to classify the paragraph into one of four fonts (from 0 to 3).

|     Font 0     |  Font 1  |  Font 2  |       Font 3       |
|----------------|----------|----------|--------------------|
|Scheherazade New|  Marhey  | Lemonada |IBM Plex Sans Arabic|


---

# Dataset Preprocessing and Model Training

## Overview
This repository contains code for loading a dataset, preprocessing the images, and training a PyTorch model using a Sequential neural network with 3 linear layers.

## Workflow

1. **Data Loading and Splitting**:
   - Loading the dataset.
   - Splitting the dataset into training and validation sets.

2. **Image Preprocessing**:
   - Removing salt and pepper noise using median blur.
   - Sharpening the image using filter2d.
   - Converting the image to binary.
   - Deskewing the image.
   - Resizing the image.

3. **Feature Extraction**:
   - Feature extraction is done using Histogram of Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT).

4. **Feature Scaling**:
   - We apply standardization to the extracted features.

5. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) is applied with n_components = 0.99 to reduce the dimensionality of the features.

6. **Model Architecture**:
   - Input layer processes feature vectors.
   - 2 hidden layers learn complex patterns with ReLU activation.
   - Output layer generates class probabilities with softmax activation.

7. **Model Training**:
   - The Sequential model is trained on the preprocessed dataset.

8. **Model Tuning**:
   - Hyperparameter tuning is performed on the validation set.
   - The best model parameters are saved after tuning.

## Usage
1. Clone the repository:

   ```bash
   git clone https://github.com/ZiaddAhmedd/Arabic-font-recognition-system.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run script:

   ```bash
   python font_recognition.py
   ```

4. Start local server using FastAPI:

   ```bash
   python -m uvicorn Deploy:app --reload
   ```

## Contact
For any inquiries, please contact:
- [Ahmed Emad](mailto:ahmed.younes01@eng-st.cu.edu.eg)
- [Hla Hany](mailto:hla.ahmed00@eng-st.cu.edu.eg)
- [Ziad Ahmed](mailto:ziad.abdelhameeed01@eng-st.cu.edu.eg)
- [Nada Tarek](mailto:nada.mohamed001@eng-st.cu.edu.eg)
---