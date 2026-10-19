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
1. **Overall Accuracy (`35.12%` across -10 dB to +18 dB) & High-SNR Feature-Set Ceiling (`48.75%` at $+14\text{ dB}$):**
   - **Low-Order Modulations (`BPSK`: 100%, `QPSK`: 86% at $+14\text{ dB}$):** Under unit-power normalization, `BPSK` has $|C_{40}| = 2.0$ ($C_{40} = -2$) and `QPSK` has $|C_{40}| = 1.0$, making them well separated. `QPSK` reaches 86% at $+14\text{ dB}$ (and 57% at $+4\text{ dB}$, while `BPSK` is 98%) due to uncompensated phase rotations under `Rayleigh3Tap` multipath fading.
   - **18-Feature Set Ceiling for $M \ge 8$ PSK:** Higher-order phase modulations (`8PSK`, `16PSK`, `32PSK`) share identical theoretical 4th- and 6th-order cumulant magnitudes ($|C_{40}| = 0, |C_{42}| = 1, |C_{60}| = 0$). Because the 18-feature set stops at 6th-order cumulants (without 8th-order $C_{80}$), these 3 classes experience intra-group confusion (contributing $\approx 3 \times \frac{1}{3} = 1.0$ correct class).
   - **Finite-Sample Variance in Dense QAM:** Over short $L=128$-sample bursts, sample estimation variance blurs the small theoretical cumulant gaps between `16QAM`, `64QAM`, and `256QAM` ($|C_{40}| \approx 0.68, 0.62, 0.61$), contributing $\approx 3 \times \frac{1}{3} = 1.0$ correct class.
   - **Mathematical Verification of the High-SNR Accuracy Ceiling:**
     $$\text{Acc}_{\text{expected}}(+14\text{ dB}) \approx \frac{1.00\text{ (BPSK)} + 0.86\text{ (QPSK)} + 1.00\text{ (3 }M\text{-PSK)} + 1.00\text{ (3 QAM)}}{8} = \frac{3.86}{8} = \mathbf{48.25\%} \quad (\text{Empirical: } \mathbf{48.75\%})$$
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
