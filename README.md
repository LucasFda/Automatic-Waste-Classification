# README for Automatic Waste Classification Project

## Project Overview
This project aims to develop an automatic waste classification system capable of identifying different types of waste materials based on images. The system utilizes a Convolutional Neural Network (CNN) trained to classify images into the following categories:

- **Clothes (C)**
- **Plastics (P)**
- **Organics (O)**
- **Metals (M)**

The primary goal is to achieve a minimum classification accuracy of 90% for each category, ensuring high precision and reliability for real-world applications.

## Folder Structure
```
PRJ
├── dataset
│   ├── train (70%)
│   ├── test  (15%)
│   └── val   (15%)
├── models
│   ├── accuracy_curve_X.png
│   ├── loss_curve_X.png
│   ├── model_X.keras
│   ├── report_X.txt
├── best_model.txt
├── build_and_train_model.py
├── check_train_test_val.py
├── predict_with_picture.py
├── prepare_test_train_val.py
├── rename_img_by_class.py
└── README.md
```

### Key Components
- **`dataset/`**: Contains subdirectories for `train`, `test`, and `val` datasets, organized into class folders (`C`, `P`, `O`, `M`).
- **`models/`**: Stores trained models (`.keras`), training plots (`accuracy_curve_`, `loss_curve_`), and classification reports (`report_`).
- **Scripts**:
  - `build_and_train_model.py`: Script for building, training, and saving the model.
  - `check_train_test_val.py`: Validates the dataset structure.
  - `predict_with_picture.py`: Predicts the class of a single image selected by the user.
  - `prepare_test_train_val.py`: Splits the dataset into train, test, and validation sets.
  - `rename_img_by_class.py`: Renames images in the dataset for uniformity.

## Setup Instructions

### 1. Prepare the Dataset
Ensure the dataset is organized with subdirectories for each class (`C`, `P`, `O`, `M`). Place the dataset in the `dataset/` directory.

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install tensorflow opencv-python matplotlib
```

### 3. Split the Dataset
Run the following command to split the dataset into training, validation, and test sets:
```bash
python prepare_test_train_val.py
```

### 4. Train the Model
Train the CNN model using the `build_and_train_model.py` script:
```bash
python build_and_train_model.py
```
This script saves the model, training plots, and classification report in the `models/` directory.

### 5. Predict with a Picture
To classify a single image, run:
```bash
python predict_with_picture.py
```
Follow the on-screen instructions to select an image. The result is saved in the `results/` directory.

## Example Usage
### Training the Model
```python
from build_and_train_model import train_model

# Train with default parameters
data_dir = "./PRJ/dataset"
train_model(data_dir=data_dir, batch_size=16, epochs=50)
```

### Predicting an Image
```python
from predict_with_picture import predict_image

predict_image("./PRJ/results/image_result.jpg")
```

## Contributing
Contributions are welcome! Feel free to submit issues or feature requests.

