# EE136IU Capstone Mini-Project - Topic 1: MLP Benchmark
**Authors:** Hai Nguyen (EEEEIU22093), Van Thao Minh (EEEEIU22097)  
**Course:** EE136IU - Machine Learning for Wireless Communication  
**Instructor:** Dr. Huynh Vo Trung Dung  

---

## 🌐 Live Web Demo Dashboard
👉 **Truy cập Web Demo trực tiếp tại:** [https://emergingproject-jjb3xffvtgrik3fhqddyzx.streamlit.app](https://emergingproject-jjb3xffvtgrik3fhqddyzx.streamlit.app)  
*(Lưu ý: Do chạy trên gói miễn phí của Streamlit Community Cloud, nếu lâu không có người truy cập app có thể tạm nghỉ — chỉ cần bấm nút **"Yes, get this app back up!"** để đánh thức trong vài giây).*

---

## 📡 Overview & Physical Insights
This repository contains the complete implementation, pre-trained weights, benchmark evaluations, and an interactive Streamlit live dashboard for **Tier 2: 3-Layer Multilayer Perceptron (MLP)** (`11,592` trainable parameters) for Automatic Modulation Classification (AMC) across 8 modulation schemes (`BPSK`, `QPSK`, `8PSK`, `16PSK`, `32PSK`, `16QAM`, `64QAM`, `256QAM`) on 18 physical domain features.

### Key Empirical & Theoretical Findings:
1. **Overall Accuracy (`35.12%` across -10 dB to +18 dB) & High-SNR Ceiling (`~48.8%` at $\ge +10\text{ dB}$):**
   - `BPSK` and `QPSK` reach **100% classification accuracy** at high SNR due to distinct 4th-order cumulants ($|C_{40}| = 1.0$) and phase entropy.
   - **Theoretical Cumulant Degeneracy for $M \ge 8$ PSK:** Higher-order phase modulations (`8PSK`, `16PSK`, `32PSK`) possess identical theoretical 4th- and 6th-order cumulant magnitudes ($|C_{40}| = 0, |C_{42}| = 1$), creating an information-theoretic identifiability ceiling on the 18 handcrafted features.
   - **Finite-Sample Variance in Dense QAM:** Over short $L=128$-sample bursts, estimation variance blurs the small theoretical cumulant gaps between `16QAM`, `64QAM`, and `256QAM` ($|C_{40}| \approx 0.68, 0.62, 0.61$).
2. **Inference Latency (Batched vs. Single-Sample):**
   - **Batched Throughput Latency ($B = 1000$):** `0.71 µs / burst` (vectorized matrix multiplication).
   - **Single-Burst Dispatch Latency ($B = 1$):** `~36 µs / burst` (including Python/PyTorch single-call dispatch overhead).

---

## 📦 Files in this Repository
- `model.py`: Standalone `ModulationMLP` (`MLP`) class definition and 1-line loader `load_pretrained_mlp()`.
- `best_mlp.pth`: Pre-trained PyTorch model checkpoint (Epoch 33, Test Accuracy: 35.12%).
- `scaler.pkl`: Fitted `StandardScaler` for the 18 physical radio features.
- `class_mapping.pkl`: Dictionary containing `'class_to_idx'`, `'idx_to_class'`, and `'features'`.
- `train_mlp.py`: Complete PyTorch training pipeline and SNR benchmarking script (guarded by `if __name__ == "__main__":`).
- `app.py`: Interactive Streamlit dashboard for real-time burst inference and benchmark visualization.
- `sample_test_bursts.csv`: Stratified test dataset (600 bursts across all 8 modulations $\times$ 15 SNR levels).
- `mlp_snr_benchmark_verified.csv` & `mlp_executive_summary.csv`: Verified performance metrics.

---

## 🚀 Quick Start for Teammates (Tested & Ready to Import)

### Option 1: One-Line Loader via `model.py`
```python
import torch
import numpy as np
from model import load_pretrained_mlp

# 1. Load model, scaler, and mapping dictionary
model, scaler, mapping = load_pretrained_mlp()
idx_to_class = mapping["idx_to_class"]
feature_cols = mapping["features"]

# 2. Predict on a 18-feature burst (shape: [1, 18])
x_raw = np.zeros((1, 18), dtype=np.float32)  # replace with actual 18 features
x_scaled = scaler.transform(x_raw)

with torch.no_grad():
    logits = model(torch.from_numpy(x_scaled.astype(np.float32)))
    probs = torch.softmax(logits, dim=-1)
    pred_idx = int(torch.argmax(probs, dim=-1).item())
    prediction = idx_to_class[pred_idx]

print("Predicted Modulation:", prediction)
```

### Option 2: Manual PyTorch Loading
```python
import pickle
import torch
from model import ModulationMLP

scaler = pickle.load(open("scaler.pkl", "rb"))
mapping = pickle.load(open("class_mapping.pkl", "rb"))

model = ModulationMLP(input_dim=18, num_classes=8)
model.load_state_dict(torch.load("best_mlp.pth", map_location="cpu"))
model.eval()

# Inference
X_scaled = scaler.transform(X_raw_18_features)
with torch.no_grad():
    logits = model(torch.tensor(X_scaled, dtype=torch.float32))
    probs = torch.softmax(logits, dim=-1)
    pred_class = mapping["idx_to_class"][int(torch.argmax(probs).item())]
```
