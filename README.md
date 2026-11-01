# AgriVision: AI-Powered Multi-Crop Disease Diagnosis & Precision Agronomy System 🌾🌿

AgriVision is an intelligent agricultural crop health diagnosis and clinical advisory platform designed to help farmers, agronomists, and researchers rapidly identify crop leaf pathologies and receive immediate, actionable therapeutic treatments.

---

## 👥 Academic Project Credits

- **Developers:** Akhilesh Korath & Jaydeep Modi
- **Academic Program:** Computer Science and Business Systems (CSBS), 3rd Year / 5th Semester
- **Department:** Center for Computer Science and Technology Department
- **Under the Mentorship of:** Dr. Vivek Sharma, Assistant Professor

---

## 🎯 Key Features & How It Works

1. **Strict Crop Isolation Engine:**
   - Evaluates leaf photos strictly within the selected crop's disease subspace, eliminating cross-crop false positives (e.g. Corn diseases will never be misdiagnosed as Apple pathologies).

2. **Multi-Model Neural Ensemble (ONNX):**
   - **`general_crops.onnx`**: SOTA MobileNet/ResNet model trained on PlantVillage benchmark (Apple, Corn, Potato, Tomato, Pepper, Soybean) achieving 98.5%+ validation accuracy.
   - **`rice_model.onnx`**: High-resolution ResNet-50 model specialized for Rice diseases (Bacterial Leaf Blight, Brown Spot, Leaf Smut, Tungro).
   - **`remaining_crops.onnx`**: Custom-trained multi-class ONNX neural network covering Wheat, Cotton, Sugarcane, Onion, Rose, and Chickpea.

3. **13 Strictly Supported Staple Crops:**
   - 🍏 **Apple** (Scab, Black Rot, Cedar Apple Rust, Healthy)
   - 🌱 **Chickpea** (Ascochyta Blight, Fusarium Wilt, Healthy)
   - 🌽 **Corn / Maize** (Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy)
   - ☁️ **Cotton** (Bacterial Blight, Curl Virus, Leaf Spot, Healthy)
   - 🧅 **Onion** (Purple Blotch, Downy Mildew, Healthy)
   - 🫑 **Pepper / Bell Pepper** (Bacterial Spot, Healthy)
   - 🥔 **Potato** (Early Blight, Late Blight, Healthy)
   - 🌾 **Rice / Paddy** (Bacterial Leaf Blight, Brown Spot, Leaf Smut, Tungro, Healthy)
   - 🌹 **Rose** (Black Spot, Powdery Mildew, Healthy)
   - 🫘 **Soybean** (Bacterial Blight, Frogeye Leaf Spot, Healthy)
   - 🎋 **Sugarcane** (Red Rot, Leaf Scald, Rust, Healthy)
   - 🍅 **Tomato** (Early Blight, Late Blight, Septoria, Yellow Leaf Curl Virus, Spider Mites, Bacterial Spot, Target Spot, Healthy)
   - 🌾 **Wheat** (Septoria, Stripe Rust, Brown Rust, Healthy)

4. **Multi-Tiered Therapeutic Advisory:**
   - **Chemical Treatments:** Specific commercial active ingredients (e.g. Mancozeb, Azoxystrobin, Chlorothalonil) with precise application dosages (g/L or ml/L).
   - **Biological & Organic Remedies:** Eco-friendly alternatives including Neem oil (Azadirachtin), *Trichoderma harzianum*, and copper bio-washes.
   - **Cultural & Preventive Care:** Canopy management, drip irrigation advice, crop rotation, and sanitization protocols.

5. **Integrated Agri-Chatbot:**
   - Agricultural chat assistant providing answers in English and Hindi for crop nutrition, fertilizer schedules, and pest mitigation.

---

## 🏗️ Project Architecture

```
Agrivision/
├── models/
│   ├── general_crops.onnx       # SOTA model for Apple, Corn, Potato, Tomato, Pepper, Soybean
│   ├── rice_model.onnx          # ResNet-50 model specialized for Rice
│   ├── remaining_crops.onnx     # Deep ONNX model for Wheat, Cotton, Sugarcane, Onion, Rose, Chickpea
│   └── remaining_classes.json   # Class labels mapping for remaining crops
├── public/
│   └── index.html               # Responsive web interface (Tailwind CSS, Lucide Icons)
├── disease_info.json            # Clinical agricultural knowledge base (symptoms, remedies, chemical dosages)
├── classes.json                 # Label registry
├── inference.py                 # Strict multi-model ONNX inference engine
├── server.js                    # Express.js REST API & AI advice server
├── package.json                 # Node.js dependencies
└── README.md                    # Project documentation
```

---

## ☁️ Deploy to Render (Cloud Hosting)

This repository is pre-configured with a production-ready `Dockerfile` and `render.yaml` for 1-click cloud deployment on [Render](https://render.com).

### Quick Steps:
1. Log in to [Render Dashboard](https://dashboard.render.com/) with GitHub.
2. Click **New +** &rarr; **Web Service**.
3. Connect your repository: `AkhileshKorath/Agrivision-`.
4. Render will automatically detect the **Docker** environment:
   - **Runtime:** `Docker`
   - **Plan:** `Free`
5. *(Optional)* Under **Environment Variables**, add:
   - `GROQ_API_KEY`: `your_key_here` (for the AI agronomy chatbot)
6. Click **Deploy Web Service** &mdash; Render will build the Docker container and give you a live HTTPS URL!

---

## 🚀 Getting Started Locally

### Prerequisites
- **Node.js** (v18 or higher)
- **Python** (v3.9 or higher)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AkhileshKorath/Agrivision-.git
   cd Agrivision-
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Install Python dependencies:**
   ```bash
   pip install onnxruntime numpy pillow
   ```

4. **Set up Environment Variables:**
   Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Add your optional LLM API key if using the chat assistant:
   ```env
   PORT=3000
   GROQ_API_KEY=your_api_key_here
   ```

5. **Start the Web Application:**
   ```bash
   npm start
   # or
   node server.js
   ```

6. **Open in Browser:**
   Visit `http://localhost:3000` to access the AgriVision dashboard.

---

## 📜 License
This project is developed for educational and academic research purposes under the Center for Computer Science and Technology Department.
