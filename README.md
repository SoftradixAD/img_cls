# 🐱🐶 Unknown Image Classifier

This project is a Convolutional Neural Network (CNN)-based image classifier that can predict whether an uploaded image is of a **cat, dog, or unknown**. The model is trained on grayscale images and achieves **75% accuracy**.

## 🚀 Features
- Upload an image through a **futuristic web interface**.
- Classifies the image as **Cat, Dog, or Unknown**.
- Flask backend for handling image processing and predictions.
- TensorFlow-based CNN model for classification.

---

## 🛠 Setup & Installation

### 1️⃣ Clone the Repository
```sh
 git clone https://github.com/SoftradixAD/img_cls.git
 cd img_cls
```0

### 2️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed, then install required packages:
```sh
pip install -r requirements.txt
```

### 3️⃣ Download Dataset
Run the scripts to download images from the internet:
```sh
python cat.py
python dog.py
python unknown.py
```

### 4️⃣ Convert Images to Grayscale
```sh
python grayscalecat.py
python grayscaledog.py
python grayscaleunknown.py
```

### 5️⃣ Prepare the Dataset
```sh
python train.py
```

### 6️⃣ Train the Model
This step trains a **CNN model** and saves it to the `saved_model/` directory:
```sh
python model.py
```

---

## 🎮 Running the Application

### 🔥 Start the Flask Backend
```sh
python app.py
```
By default, the Flask server runs at `http://127.0.0.1:5000/`

### 🌐 Open the Frontend
Simply open `index.html` in your browser and upload an image to classify.

---

## 🖼️ User Interface Preview
The web interface provides an intuitive image upload system with futuristic styling, allowing users to easily upload an image and view the classification results.

![Classifier UI Preview](assets/ui_preview.png) *(Replace with actual screenshot)*

---

## 📌 API Endpoint
### **POST /predict**
Uploads an image and returns a classification.
#### **Request**:
- **Content-Type**: `multipart/form-data`
- **Form Data Parameter**: `file`

#### **Response**:
```json
{
  "prediction": "cat",
  "confidence": 0.87
}
```

---

## 🤖 Model Details
- Architecture: **Convolutional Neural Network (CNN)**
- Input Size: **256x256 (Grayscale)**
- Accuracy: **75%**
- Framework: **TensorFlow / Keras**

---

## 📜 License
This project is licensed under the MIT License.

---

## 🤝 Contributing
Pull requests are welcome! If you find an issue, feel free to create one or submit a PR.

---

## 🌟 Acknowledgments
- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)

---

## 📬 Contact
For queries or collaborations, reach out via [your.email@example.com](mailto:your.email@example.com).

---

⭐ **If you like this project, don't forget to star it!** ⭐

