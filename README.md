
# Smartel_Farmer_Life_Line
## Rice Disease Classification using Transfer Learning

### Project Overview
In this project, I've implemented a Convolutional Neural Network (CNN) using transfer learning to classify rice plant diseases. The model is based on the VGG16 architecture and is trained to identify four types of rice diseases:

1. Bacterial Blight Disease
2. Blast Disease
3. Brown Spot Disease
4. False Smut Disease

### Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Optimization Techniques](#optimization-techniques)
- [Training](#training)
- [Evaluation](#evaluation)
- [Error Analysis](#error-analysis)
- [Results](#results)
- [Future Improvements](#future-improvements)

### Prerequisites
To run this project, you'll need:

- Python 3.x
- Jupyter Notebook
- Required libraries (see Installation)

### Installation
To set up the project environment, run the following command:
```bash
pip install opencv-python numpy matplotlib seaborn scikit-learn tensorflow
```

### Dataset
The dataset is stored in the `./Rice_Diseases` directory. It contains images of rice plants affected by four different diseases. The dataset is organized into subdirectories, each representing a disease category.

### Project Structure
This project is structured as a single Jupyter notebook that contains all the code for data loading, preprocessing, model definition, training, and evaluation.

### Usage
To use this project:

1. Open the Jupyter notebook in your preferred environment.
2. Run all cells in order to execute the entire pipeline.
3. The notebook will load the data, preprocess images, train the model, and evaluate its performance.

### Model Architecture
For the model, I've used transfer learning with the VGG16 architecture:

- **Base**: Pre-trained VGG16 (weights from ImageNet, without top layers)
- **Custom top layers**:
  - Flatten layer
  - Dense layer (512 units) with ReLU activation and L2 regularization
  - Output Dense layer (4 units) with softmax activation

### Optimization Techniques
1. **Transfer Learning**: Used a pre-trained VGG16 model to leverage knowledge from general image features.
2. **Data Augmentation**: Applied transformations such as rotation, shifting, shearing, zooming, and flipping to improve model generalization.
3. **L2 Regularization**: Applied to the Dense layer to prevent overfitting.
4. **Dropout**: Added Dropout (50%) to prevent overfitting.
5. **Early Stopping**: Monitored validation loss and stopped training when no improvement was detected.
6. **Learning Rate Reduction**: Reduced the learning rate by 20% when performance plateaued.

### Training
The model was trained using the following configuration:

- **Optimizer**: Adam
- **Loss function**: Categorical Cross-entropy
- **Metrics**: Accuracy
- **Data Augmentation**: Rotation, width/height shift, shear, zoom, and horizontal flip
- **Callbacks**: EarlyStopping and ReduceLROnPlateau

### Evaluation
The evaluation was done using:

- Test set accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

### Error Analysis
The model was tested across three configurations: the vanilla model (no regularization), L2 regularization, and L1 regularization. Below are the test accuracy and F1 scores for each model:

- **Vanilla Model**: 
  - Test Accuracy: 94.12%
  - Weighted F1 Score: 0.9412
- **L2 Regularization Model**: 
  - Test Accuracy: 94.12%
  - Weighted F1 Score: 0.9412
- **L1 Regularization Model**: 
  - Test Accuracy: 91.18%
  - Weighted F1 Score: 0.9106

#### Based on the training and validation loss graphs:
1. **Vanilla Model**: The first graph shows that the vanilla model converged well, with some fluctuations in validation loss but minimal overfitting. The test accuracy and F1 score are quite high, indicating good performance.
   
2. **L2 Regularization Model**: The second graph shows that the L2 model performed similarly to the vanilla model, but with smoother convergence and slightly better validation loss behavior. It effectively controlled overfitting while maintaining high accuracy and F1 score.

3. **L1 Regularization Model**: The third graph shows the L1 model had the smoothest convergence of all, especially in the early epochs. However, it resulted in a slight decrease in test accuracy and F1 score compared to the other models. This model performed well in terms of regularization but did not generalize as effectively as the vanilla and L2 models.

#### Which Model is Best?
While all three models performed well, the **L2 regularization model** stands out as the best overall. It balanced performance and generalization, achieving high accuracy and F1 score, with smoother convergence. The L1 model, although having smoother loss reduction, showed lower accuracy and F1 scores, making it less ideal for this task.

### Results
The model achieved the following performance on the test set:

- **Test Accuracy**: 94.12% (Vanilla and L2 Regularization)
- **Class-wise performance**:
  - **Bacterial Blight Disease**: Precision 1.00, Recall 0.91
  - **Blast Disease**: Precision 0.75, Recall 0.75
  - **Brown Spot Disease**: Precision 0.91, Recall 0.91
  - **False Smut Disease**: Precision 0.89, Recall 1.00

### Future Improvements
1. **More Data**: Collect more diverse training data, especially for Blast Disease and Brown Spot Disease.
2. **Cross-validation**: Implement cross-validation to get a more robust estimate of model performance.
3. **Model Architectures**: Experiment with different model architectures to improve the ability to distinguish between similar diseases.
4. **Hyperparameter Tuning**: Fine-tune hyperparameters to reduce overfitting.
5. **Advanced Data Augmentation**: Apply data augmentation techniques like color jittering for better generalization.
6. **Ensemble Methods**: Explore ensemble methods to combine predictions from multiple models.
7. **Attention Mechanisms**: Investigate the use of attention mechanisms to help the model focus on relevant parts of the images.
