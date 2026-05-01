# PULSE - AI-Powered Emergency Triage Risk Predictor

![Accuracy](https://img.shields.io/badge/Accuracy-94.27%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)

## Overview

PULSE is an AI-powered web application that predicts emergency triage risk levels using 8 simple vital signs. It achieves **94.27% accuracy** and helps ER doctors identify high-risk patients within 1 minute of arrival.

## Features

- ✅ **94.27% Accuracy** - Trained on 18,000+ patients
- ✅ **8 Simple Inputs** - Age, heart rate, blood pressure, oxygen, temperature, pain level, chronic diseases, arrival mode
- ✅ **3 Languages** - English, Arabic (RTL), French
- ✅ **Prediction History** - Every prediction saved to database
- ✅ **Offline Ready** - Works without internet
- ✅ **Beautiful UI** - White & Blue theme

## Technology Stack

| Component | Technology |
|-----------|------------|
| AI Model | XGBoost |
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Training Data | 18,000+ patients |

## Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PULSE.git
cd PULSE
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```
## 3. Train the model
```bash
python train_model.py
```
## 4. Run the application
```bash
python api.py
```
## 5. Open your browser

Go to: http://localhost:5000

## Author:

Grabsi Khadidja
