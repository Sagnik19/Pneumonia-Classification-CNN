# 🫁 Pneumonia Classification using CNN

## 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) to classify Chest X-Ray images as:

* NORMAL
* PNEUMONIA

The model was built using TensorFlow and Keras and trained on the Chest X-Ray Pneumonia dataset from Kaggle.

---

## 🚀 Features

* Chest X-Ray Image Classification
* Image Preprocessing & Augmentation
* CNN Deep Learning Model
* Model Evaluation
* Streamlit Web Application
* Binary Classification

---

## 📂 Project Structure

```text
Pneumonia-Classification-CNN/
│
├── app.py
├── model.py
├── notebooks/
│   ├── 01_pneumonia_EDA.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_cnn_model.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── models/
│   └── trained/
│       └── pneumonia_cnn_final.keras
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

Dataset: Chest X-Ray Pneumonia Dataset

Classes:

* NORMAL
* PNEUMONIA

Dataset Size:

* Training Images: 5,216
* Test Images: 624

---

## 🧠 Model Architecture

* Conv2D
* MaxPooling2D
* Conv2D
* MaxPooling2D
* Conv2D
* MaxPooling2D
* Flatten
* Dense
* Dropout
* Dense (Output Layer)

Activation Function:

* ReLU
* Sigmoid

Loss Function:

* Binary Crossentropy

Optimizer:

* Adam

---

## 📈 Training Results

Training Accuracy:

* Above 90%

Validation Accuracy:

* Up to 87.5%

---

## 🖥️ Streamlit Application

Run:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## ⚙️ Installation

Clone Repository:

```bash
git clone <repository-url>
```

Create Virtual Environment:

```bash
python -m venv venv
```

Activate Environment:

```bash
venv\Scripts\activate
```

Install Requirements:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Streamlit
* Scikit-Learn

---

## 👨‍💻 Author

Sagnik Das

Data Science Student | Machine Learning & Deep Learning Enthusiast
