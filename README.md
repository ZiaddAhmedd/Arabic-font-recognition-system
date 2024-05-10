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
   - First, we load the dataset.
   - Then, we split the dataset into training and validation sets.

2. **Image Preprocessing**:
   - Preprocessing steps include:
     - Removing salt and pepper noise using median blur.
     - Sharpening the image using filter2d.
     - Converting the image to binary.
     - Deskewing the image.
     - Resizing the image.

3. **Feature Extraction**:
   - Feature extraction is done using Histogram of Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT).

4. **Feature Scaling**:
   - We apply standard scaling to the extracted features.

5. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) is applied with n_components = 0.99 to reduce the dimensionality of the features.

6. **Model Architecture**:
   - The model architecture consists of a PyTorch Sequential model with 3 linear layers:
     1. Linear layer with input dimension `self.input_dim` and hidden dimension `self.hidden_dim1`, followed by ReLU activation.
     2. Linear layer with hidden dimension `self.hidden_dim1` and `self.hidden_dim2`, followed by ReLU activation.
     3. Linear layer with hidden dimension `self.hidden_dim2` and output dimension `self.output_dim`.

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

3. Run the preprocessing and training script:

   ```bash
   python font_recognition.py
   ```

## Contact
For any inquiries, please contact:
- [Ahmed Emad](mailto:)
- [Hla Hany](mailto:hla.ahmed00@eng-st.cu.edu.eg)
- [Ziad Ahmed](mailto:)

---