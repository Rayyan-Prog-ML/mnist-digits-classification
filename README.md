# 🧠 MNIST Digit Classifier

## 📌 Overview  
This project implements a **handwritten digit classifier** using the **MNIST dataset**. The model is built with **TensorFlow** and **Keras**, leveraging **deep learning techniques** to recognize digits (0-9) from images.  

## 🚀 Features  
- **Deep Learning with TensorFlow & Keras** – Uses a neural network for accurate classification.  
- **Preprocessing with Scikit-Learn** – Normalization and reshaping for model training.  
- **Performance Evaluation** – Metrics like accuracy and confusion matrix are used for assessment.  
- **Handwritten Digit Recognition** – Trained on the MNIST dataset to classify digits (0-9).  

## 📂 Dataset  
The **MNIST dataset** consists of **60,000** training images and **10,000** test images of handwritten digits (28x28 pixels, grayscale).  

## 🛠️ Technologies Used  
- **Python** 🐍  
- **TensorFlow & Keras** 🔥  
- **Scikit-Learn** 🎯  
- **NumPy & Pandas** 🏗️  
- **Matplotlib & Seaborn** 📊  

## ⚙️ Installation & Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```  
2. Install dependencies:  
   ```bash
   pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
   ```  
3. Run the Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```  
4. Open `mnist_digit_classifier.ipynb` and execute the cells.  

## 🎯 Usage  
- The model is trained using the MNIST dataset and predicts handwritten digits from test images.  
- You can visualize training accuracy, loss curves, and confusion matrix after model evaluation.  
- Modify the model architecture in the notebook for experimentation.  

## 📊 Model Performance  
- Achieved **high accuracy** on the test dataset.
- **Training Accuracy** 99.39%
- **Validation Accuracy** 98.15%  


## 📌 Sample Prediction  
```python
# Predict a sample image
import matplotlib.pyplot as plt
import numpy as np

index = 5  # Change index for different predictions
image = X_test[index].reshape(28, 28)
plt.imshow(image, cmap="gray")
plt.title(f"Predicted: {model.predict_classes(X_test[index].reshape(1, 28, 28, 1))[0]}")
plt.show()
```  

## 📜 License  
This project is licensed under the **MIT License** – feel free to use and modify it!  

## 💡 Future Improvements  
✅ Implement **Convolutional Neural Networks (CNNs)** for higher accuracy  
✅ Add **Data Augmentation** to improve generalization  
✅ Convert model to **TFLite** for mobile deployment  

## 📬 Connect with Me  
If you like this project, give it a ⭐ on **GitHub**! Feel free to connect on **LinkedIn** or contribute to the project. 🚀  
