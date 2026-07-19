# 🌊 Rock vs Mine Prediction System

An end-to-end Machine Learning web application built using **Streamlit** and **Scikit-Learn**. The application utilizes a **Logistic Regression** model trained on the Connectionist Bench (Sonar) dataset to classify underwater objects as either a harmless **Rock (🪨)** or a dangerous **Mine (💣)** based on 60 sonar signal frequencies.

---

## 🚀 Features

- **Interactive User Interface:** A modern and responsive Streamlit web application.
- **60-Frequency Input Form:** Input numerical values for all 60 distinct sonar signal features.
- **Real-Time Prediction:** Immediate classification of the input signal as Rock or Mine.
- **Confidence Visualization:** Provides prediction confidence scores (probabilities) represented visually with progress bars.
- **Self-Healing Model Pipeline:** Automatically trains a new Logistic Regression model if no pre-trained model checkpoint (`sonar_model.pkl`) is detected locally.

---

## 📁 Repository Structure

```markdown
├── app.py                            # Streamlit Web Application & Prediction Pipeline
├── sonar data.csv                    # Sonar dataset used for training/validation
├── Rock_vs_Mine_Prediction.ipynb     # Jupyter Notebook containing model exploration & training tests
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git ignore file
```

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.x
- **Frontend Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Persistence:** [Joblib](https://joblib.readthedocs.io/)

---

## ⚙️ Quick Start Guide

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/adminsidra/Rock-vs-Mine-prediction-sys.git
cd Rock-vs-Mine-prediction-sys
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
This will start the Streamlit server and automatically open the application in your default web browser (usually at `http://localhost:8501`).

---

## 📊 Dataset Information

The model is trained on the **Sonar, Mines vs. Rocks Dataset** containing 208 instances:
- **Features:** 60 columns representing energy in particular frequency bands, integrated over a certain period of time (values range between 0.0 and 1.0).
- **Target:** Class label `R` (Rock) or `M` (Mine).

---

## 💡 How It Works

1. When you run `app.py`, the system checks for a saved model file `sonar_model.pkl`.
2. If the model file is not found, the application automatically loads `sonar data.csv`, trains a **Logistic Regression** model, saves the trained model to `sonar_model.pkl`, and uses it.
3. Users input 60 sonar values through the UI form and click **🔍 Predict Object**.
4. The application processes the input data, shapes it, and uses the model to predict the class and compute the confidence score.
