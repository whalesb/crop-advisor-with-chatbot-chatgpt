# AgroSense_AI

# 🌱 AgroSense-AI: The Crop Compatibility Engine

AgroSense-AI is an intelligent crop recommendation and yield prediction system designed to assist farmers, agricultural advisors, and researchers in making data-driven decisions. It evaluates environmental conditions and soil nutrients to recommend the most compatible crops and estimate potential yield, maximizing productivity while minimizing waste.

## 📌 Problem Statement

Farmers and individuals lack automated systems to match crops with their environment, leading to poor yields and wasted resources. AgroSense-AI solves this by using machine learning to recommend crops and predict yield based on environmental and soil data.

---

## 🚀 Features

- ✅ **Crop Recommendation System**  
  Uses machine learning (Random Forest Classifier) to recommend the best-suited crops based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - pH level
  - Temperature
  - Humidity
  - Rainfall
  - Soil Moisture

- 📊 **Yield Prediction (Beta)**  
  Estimates potential crop yield using a regression model based on real and synthetically augmented datasets.

- 🧠 **Interactive Interface**  
  Built with **Streamlit** for easy input and real-time predictions using sliders and dropdowns.

- 🌐 **Explainability**  
  Feature importance visualization to show which factors influence recommendations.

---

## 🧪 Tech Stack

- **Python**
- **Pandas & NumPy** – Data processing
- **Scikit-learn** – ML algorithms
- **Matplotlib & Seaborn** – Visualization
- **Streamlit** – User Interface
- **Jupyter Notebook** – Data exploration and development

---

## 📁 File Structure


---

## 📊 How It Works

1. **Input Data**: User enters soil and environmental data via sliders.
2. **Crop Prediction**: ML model predicts best-fit crops.
3. **Yield Prediction (Optional)**: System estimates expected yield based on all inputs.
4. **Output**: Recommended crop and yield estimate with explanations.

---

## 📦 Installation & Run

### 🔧 Requirements
```bash
pip install -r requirements.txt
