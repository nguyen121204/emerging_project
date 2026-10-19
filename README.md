# EE136IU Capstone Mini-Project - Topic 1: MLP Benchmark
**Author:** Hai Nguyen (Student ID: EEEEIU22093)  
**Course:** EE136IU - Machine Learning for Wireless Communication  
**Instructor:** Dr. Huynh Vo Trung Dung  

---

## 🌐 Live Web Demo Dashboard (Online 24/7)
👉 **Truy cập Web Demo trực tiếp tại:** [https://emergingproject-jjb3xffvtgrik3fhqddyzx.streamlit.app](https://emergingproject-jjb3xffvtgrik3fhqddyzx.streamlit.app)  
*(Mọi người có thể mở link trên để trực tiếp bốc mẫu tín hiệu I/Q, chọn mức SNR từ -10 dB đến +18 dB và xem mạng MLP dự đoán thời gian thực).*

---

## 📡 Overview
This repository contains the complete implementation, pre-trained weights, benchmark evaluations, and an interactive Streamlit live dashboard for **Tier 2: 3-Layer Multilayer Perceptron (MLP)** for Automatic Modulation Classification (AMC) across 8 modulation schemes (`BPSK`, `QPSK`, `8PSK`, `16PSK`, `32PSK`, `16QAM`, `64QAM`, `256QAM`) on 18 physical domain features.

## 📦 Files in this Repository
- `best_mlp.pth`: Pre-trained PyTorch model checkpoint (Epoch 33, Test Accuracy: 35.12%).
- `scaler.pkl`: Fitted `StandardScaler` for the 18 physical radio features.
- `class_mapping.pkl`: Mapping dictionary between numerical class indices and modulation names.
- `train_mlp.py`: Complete PyTorch training pipeline, architecture, and SNR benchmarking logic.
- `app.py`: Interactive Streamlit dashboard for real-time live burst inference.
- `sample_test_bursts.csv`: Calibrated test bursts across 15 SNR levels (-10 dB to +18 dB).
- `mlp_snr_benchmark_verified.csv`: Detailed SNR-conditioned Accuracy and Macro-F1 scores.
- `mlp_executive_summary.csv`: Overall performance summary and latency metrics.

---

## 🚀 Quick Start for Teammates

### How to Load the Pre-Trained Model in Python:
```python
import torch, pickle
from train_mlp import MLP # or ModulationMLP

# 1. Load scaler & label mapping
scaler = pickle.load(open('scaler.pkl', 'rb'))
class_mapping = pickle.load(open('class_mapping.pkl', 'rb'))

# 2. Load trained weights
model = MLP()
model.load_state_dict(torch.load('best_mlp.pth', map_location='cpu'))
model.eval()

# 3. Predict on 18 physical features:
X_scaled = scaler.transform(X_raw_18_features)
with torch.no_grad():
    logits = model(torch.tensor(X_scaled, dtype=torch.float32))
    probs = torch.softmax(logits, dim=-1)
    prediction = class_mapping[torch.argmax(probs).item()]
```
