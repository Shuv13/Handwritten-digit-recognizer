# 🖐️ Handwritten Digit Recognizer with Python & Deep Learning

A simple deep learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) drawn on a GUI canvas. Built using the MNIST dataset, TensorFlow, Keras, and Tkinter.

---

## 🚀 Features

- Trains a CNN model on the MNIST dataset
- Interactive GUI for drawing digits
- Real-time prediction with confidence score
- Built entirely in Python

---

## 📂 Project Structure

```
.
├── train_mnist.py            # Model training script
├── gui_digit_recognizer.py   # GUI app to draw and predict digits
├── mnist.h5                  # Saved trained model (generated after training)
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository (or create a project folder)

```bash
git clone https://github.com/Shuv13/Handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

### 1. Train the CNN model

```bash
python train_mnist.py
```

This will train the model on the MNIST dataset and save it as `mnist.h5`.

### 2. Run the GUI application

```bash
python gui_digit_recognizer.py
```

### 3. Draw & Recognize

- Use your mouse to draw a digit in the white canvas area.
- Click **"Recognise"** to get a prediction and confidence score.
- Click **"Clear"** to reset the canvas.

---

## 🧠 Model Architecture

- **Conv2D** → **Conv2D** → **MaxPooling2D** → **Dropout**
- **Flatten** → **Dense** → **Dropout** → **Dense(10)** with **softmax** activation

---

## 📦 Requirements

- Python 3.8–3.12
- TensorFlow ≥ 2.12
- Keras
- Pillow
- NumPy

See `requirements.txt` for full details.

---

## 📝 Notes

- Ensure the `mnist.h5` model file is generated by running `train_mnist.py` before launching the GUI.
- The GUI requires a working installation of Tkinter (included with standard Python installations).
