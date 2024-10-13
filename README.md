
# Smartel_Farmer_Life_Line
Rice Disease Classification using Transfer Learning

## Project Overview
In this project, I've implemented a Convolutional Neural Network (CNN) using transfer learning to classify rice plant diseases. My model is based on the VGG16 architecture and is trained to identify four types of rice diseases:

- Bacterial Blight Disease
- Blast Disease
- Brown Spot Disease
- False Smut Disease

## Table of Contents
1. Prerequisites
2. Installation
3. Dataset
4. Project Structure
5. Usage
6. Model Architecture
7. Optimization Techniques
8. Training
9. Evaluation
10. Error Analysis
11. Results
12. Future Improvements

## Prerequisites
To run this project, you'll need:

- Python 3.x
- Jupyter Notebook
- Required libraries (see Installation)

## Installation
To set up the project environment, run the following commands:

```bash
pip install opencv-python numpy matplotlib seaborn scikit-learn tensorflow
```

## Dataset
I've used a dataset stored in the `./Rice_Diseases` directory. It contains images of rice plants affected by four different diseases. The dataset is organized into subdirectories, each representing a disease category.

## Project Structure
This project is structured as a single Jupyter notebook that contains all the code for data loading, preprocessing, model definition, training, and evaluation.

## Usage
To use this project:

1. Open the Jupyter notebook in your preferred environment.
2. Run all cells in order to execute the entire pipeline. The notebook will load the data, preprocess images, train the model, and evaluate its performance.

## Model Architecture
For this project, I used transfer learning with the VGG16 architecture:

- **Base**: Pre-trained VGG16 (weights from ImageNet, without top layers)
- **Custom top layers**: 
  - Flatten layer
  - Dense layer (512 units) with ReLU activation and L2 regularization
  - Dropout layer (50% rate)
  - Output Dense layer (4 units) with softmax activation

## Optimization Techniques
### 1. Transfer Learning
**Principle**: Transfer learning leverages knowledge gained from solving one problem and applies it to a related problem. For this project, I used the pre-trained VGG16 model.

**Relevance**: By using a model pre-trained on ImageNet, I was able to take advantage of general image features already learned. This allowed my model to focus on learning disease-specific features more quickly and effectively.

**Parameters**:
- `weights='imagenet'`: Loads VGG16 pre-trained on ImageNet.
- `include_top=False`: Excludes the fully connected layers, enabling me to add my custom layers.

**Justification**: This configuration allowed me to benefit from VGG16's strong feature extraction abilities while still customizing the model for the rice disease classification task.

### 2. Data Augmentation
**Principle**: Data augmentation artificially expands the training dataset by applying transformations to existing images, helping prevent overfitting.

**Relevance**: Given my limited dataset, data augmentation was crucial for enhancing the model's generalization ability.

**Parameters**:
- `rotation_range=40`: Rotates images up to 40 degrees.
- `width_shift_range=0.2`, `height_shift_range=0.2`: Randomly shifts images horizontally and vertically by up to 20%.
- `shear_range=0.2`, `zoom_range=0.2`: Applies random shearing and zooming.
- `horizontal_flip=True`: Flips images horizontally.

**Justification**: These values introduce variation while keeping the integrity of the rice disease images. They’re based on common practices for image classification tasks.

### 3. L2 Regularization
**Principle**: L2 regularization adds a penalty for large weights, promoting simpler models.

**Relevance**: L2 regularization helped to reduce overfitting by discouraging the model from assigning high weights to any particular feature.

**Parameters**:
- `kernel_regularizer=regularizers.l2(0.001)`: Applies L2 regularization with a factor of 0.001 to the dense layers.

**Justification**: The chosen value (0.001) strikes a balance between over-regularizing and allowing the model to capture useful patterns.

### 4. Dropout
**Principle**: Dropout randomly disables a fraction of neurons during training, preventing overfitting.

**Relevance**: It served as a regularization technique by preventing co-adaptation of neurons.

**Parameters**:
- `Dropout(0.5)`: 50% of neurons were dropped during training.

**Justification**: A dropout rate of 0.5 is a well-established value for balancing regularization and information retention.

### 5. Early Stopping
**Principle**: Early stopping halts training when the model's performance on the validation set stops improving, thus preventing overfitting.

**Parameters**:
- `monitor='val_loss'`, `patience=10`, `restore_best_weights=True`

**Justification**: These settings ensured that the model wouldn’t train for too long, but still allowed for sufficient improvement.

### 6. Learning Rate Reduction
**Principle**: Reducing the learning rate helps the model converge to a better solution in later training stages.

**Parameters**:
- `monitor='val_loss'`, `factor=0.2`, `patience=5`, `min_lr=0.00001`

**Justification**: These settings allow for gradual learning rate reductions, helping the model fine-tune its performance.

## Training
I trained the model using the following configuration:

- **Optimizer**: Adam
- **Loss function**: Categorical Cross-entropy
- **Metrics**: Accuracy
- **Data Augmentation**: Applied as detailed above
- **Callbacks**: Early stopping and learning rate reduction

## Evaluation
I evaluated the model using:

- **Test Accuracy**
- **Confusion Matrix**
- **Classification Report** (precision, recall, F1-score)

## Error Analysis
### Vanilla Model:
- **Test Accuracy**: 94.12%
- **Weighted F1 Score**: 0.9412
The vanilla model showed good generalization with smooth loss convergence but indicated potential overfitting.

### L2 Regularization Model:
- **Test Accuracy**: 94.12%
- **Weighted F1 Score**: 0.9412
L2 regularization performed similarly to the vanilla model, providing smoother convergence and slightly better generalization.

### L1 Regularization Model:
- **Test Accuracy**: 91.18%
- **Weighted F1 Score**: 0.9106
L1 regularization simplified the model but resulted in a small decrease in accuracy. It, however, converged smoothly.

### Conclusion:
The **L2 model** emerged as the best option, balancing smooth convergence and top performance, making it suitable for generalization while keeping the complexity in check.

## Results
- **Test Accuracy**: 94.12%
- **Weighted F1 Score**: 0.9412

## Future Improvements
- Collect more data for better model generalization.
- Experiment with different model architectures or advanced regularization techniques.
- Apply cross-validation to ensure robust performance estimates.
