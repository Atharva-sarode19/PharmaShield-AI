# 🛡️ PharmaShield AI

### AI-Powered Counterfeit Medicine Packaging Screening

PharmaShield AI is an AI-assisted screening prototype that analyzes medicine
packaging images and predicts whether the package appears **Fake or Real**.

The system uses **EfficientNet-B0** for image classification and **Grad-CAM**
to provide a visual explanation of the model's prediction.

> **Learn the Genuine. Detect the Unknown. Explain the Evidence.**

---

## 🚨 Problem

Counterfeit medicine packaging can closely resemble genuine packaging, making
manual identification difficult.

Small differences in:

- Logo
- Color
- Typography
- Layout
- Packaging design

may be difficult to notice.

PharmaShield provides a first-level AI-assisted screening approach based on
visual analysis.

---

## 💡 Current Solution

The current prototype provides:

- 📷 Medicine package image upload
- 🤖 Fake / Real image classification
- 📊 Model confidence
- 🚨 Baseline risk score
- 💡 Grad-CAM visual explanation
- 🌐 Streamlit web application

### Current Workflow

```text
Upload Package Image
        ↓
Image Preprocessing
        ↓
EfficientNet-B0
        ↓
Fake / Real Prediction
        ↓
Confidence + Risk Score
        ↓
Grad-CAM Explanation
