
# 🧠 Kidney Stone Detection

A sleek AI-powered web application for detecting kidney stones in CT images.  
Upload your scan, and our CNN model will tell you **Stone** or **Non-stone** with confidence—complete with interactive visuals.


## ✨ Features

- 🔍 Single-image upload & instant prediction  
- 📊 Confidence bars showing “Stone” vs “Non-stone”  
- 🔄 Skeleton loader while waiting  
- 📥 High-res download of results  
- 📤 Shareable result via Web Share API or clipboard  



## 🚀 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Bander03/kidney_stone_detection.git
   cd kidney_stone_detection
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**

   ```bash
   python app.py
   ```

4. **Open in browser**

   ```
   http://127.0.0.1:5000
   ```

---

## 📁 Project Structure

```text
kidney_stone_detection/
├── app.py
├── model.py
├── models/
│   └── kidney_stone_cnn.pth
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── real_example.html
│   └── docs.html
├── static/
│   └── img/
│       ├── preview-upload.png
│       └── preview-result.png
└── README.md
```

---

## 🧪 API Endpoints

| Method | Endpoint        | Description                 |
| ------ | --------------- | --------------------------- |
| GET    | `/`             | Landing page                |
| GET    | `/docs`         | Documentation page          |
| GET    | `/real_example` | Upload & predict UI         |
| POST   | `/predict`      | Receive image & return JSON |

---

## 👨‍💻 Developers

* **Bander Sidiq** – AI & Backend
* **Nawzad Rasul** – AI & Frontend




