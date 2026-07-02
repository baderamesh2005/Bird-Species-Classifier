# 🐦 Bird Species Classifier

A deep learning-based web application that identifies bird species from images using a Convolutional Neural Network (CNN). Users can upload a bird image or capture one in real time, and the trained model predicts the bird species.

---

## 📌 Features

- Upload bird images for classification.
- Real-time image capture support.
- Predicts bird species using a trained deep learning model.
- Fast and accurate inference.
- Simple Streamlit-based user interface.

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Deep Learning
- PyTorch
- Torchvision
- NumPy
- Pillow

---

## 📂 Project Structure

```
Bird-Species-Classifier/
│
├── data/                     # Dataset
├── images_to_predict/        # Sample test images
├── app_streamlit.py          # Streamlit application
├── predict.py                # Prediction script
├── train.py                  # Model training
├── model.pth                 # Trained model
├── labels.json               # Bird class labels
├── requirements.txt          # Dependencies
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/baderamesh2005/Bird-Species-Classifier.git
```

### Navigate to the project

```bash
cd Bird-Species-Classifier
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app_streamlit.py
```

After running the command, open the URL displayed in your terminal (typically `http://localhost:8501`).

---

## 🧠 Model Information

- Framework: PyTorch
- Model: Convolutional Neural Network (CNN)
- Input: Bird Image
- Output: Predicted Bird Species

---

## 📷 How It Works

1. Upload a bird image or capture one using your camera.
2. The image is preprocessed.
3. The trained CNN model extracts features and predicts the bird species.
4. The predicted bird name is displayed.

---

## 📊 Applications

- Bird species identification
- Wildlife conservation
- Biodiversity monitoring
- Educational purposes
- Nature photography assistance

---

## 📈 Future Enhancements

- Support more bird species.
- Improve prediction accuracy with larger datasets.
- Display prediction confidence scores.
- Deploy the application on Streamlit Cloud.
- Add bird information such as habitat, diet, and distribution.

---

## 📸 Sample Predictions

| Input Image | Predicted Species |
|-------------|-------------------|
| Sparrow | Sparrow |
| Crow | Crow |
| Peacock | Peacock |
| Parrot | Parrot |

---

## 👨‍💻 Author

**Ramesh Bade**

GitHub: https://github.com/baderamesh2005

---

## 📄 License

This project is intended for educational and research purposes.
