"""
EE136IU Capstone Mini-Project - Topic 1: Comprehensive Model Benchmark
Interactive Live Demo Dashboard
Author: Hai Nguyen (EE136IU)
"""

import os
import sys
import time
import zipfile
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="EE136IU AMC Live Demo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "features_train.zip") if os.path.exists(os.path.join(BASE_DIR, "features_train.zip")) else r"C:\THIS PC\My PC\Emerging\New folder\features_train.zip"

# CSS tùy chỉnh giao diện học thuật Clean White Theme
st.markdown("""
<style>
    .main-title {
        font-family: 'Manrope', sans-serif;
        color: var(--text-color, #FFFFFF);
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #0284C7;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .success-badge {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    .error-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Định nghĩa Model & Tải dữ liệu Cache
# -------------------------------------------------------------
class ModulationMLP(nn.Module):
    def __init__(self, input_dim=18, num_classes=8):
        super(ModulationMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_assets():
    with open(os.path.join(BASE_DIR, 'class_mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)
    classes = sorted(mapping['class_to_idx'].keys())
    class_to_idx = mapping['class_to_idx']
    idx_to_class = mapping['idx_to_class']
    feature_cols = mapping['features']
    
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    model = ModulationMLP(len(feature_cols), len(classes))
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'best_mlp.pth'), map_location='cpu'))
    model.eval()
    
    return classes, class_to_idx, idx_to_class, feature_cols, scaler, model

@st.cache_data
def load_test_data():
    csv_path = os.path.join(BASE_DIR, 'sample_test_bursts.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open('features_test.csv') as f:
            return pd.read_csv(f)

classes, class_to_idx, idx_to_class, feature_cols, scaler, model = load_assets()
df_test = load_test_data()

# -------------------------------------------------------------
# 3. Sidebar: Điều khiển Live Inference
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=70)
st.sidebar.markdown("### 🎛️ Bảng điều khiển tín hiệu")

# 1. Chọn loại điều chế cụ thể muốn kiểm tra
mod_options = ["Tất cả các loại (Ngẫu nhiên)"] + classes
selected_mod = st.sidebar.selectbox(
    "🎯 Chọn loại sóng muốn test:",
    options=mod_options,
    index=0
)

# 2. Chọn mức SNR
snr_options = sorted(df_test['snr_db'].unique())
selected_snr = st.sidebar.select_slider(
    "📶 Chọn mức SNR kiểm tra (dB):",
    options=snr_options,
    value=4
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ Chọn nhanh chế độ SNR:")
col_b1, col_b2, col_b3 = st.sidebar.columns(3)
if col_b1.button("-4 dB"):
    selected_snr = -4
if col_b2.button("+4 dB"):
    selected_snr = 4
if col_b3.button("+14 dB"):
    selected_snr = 14

# Lọc dữ liệu theo cả SNR và Modulation đã chọn
df_subset = df_test[df_test['snr_db'] == selected_snr]
if selected_mod != "Tất cả các loại (Ngẫu nhiên)":
    df_subset = df_subset[df_subset['modulation'] == selected_mod]

df_subset = df_subset.reset_index(drop=True)

if len(df_subset) == 0:
    st.warning("Không tìm thấy mẫu phù hợp với lựa chọn này!")
    st.stop()

if 'sample_idx' not in st.session_state or st.session_state.sample_idx >= len(df_subset):
    st.session_state.sample_idx = 0

if st.sidebar.button("🎲 Bốc mẫu tín hiệu ngẫu nhiên", use_container_width=True):
    st.session_state.sample_idx = np.random.randint(0, len(df_subset))

sample_idx = st.sidebar.slider("Chỉ số mẫu trong tập lọc:", 0, len(df_subset) - 1, st.session_state.sample_idx)
st.session_state.sample_idx = sample_idx

current_sample = df_subset.iloc[sample_idx]
true_label = current_sample['modulation']

# -------------------------------------------------------------
# 4. Giao diện chính (Tabs: Live Inference & Benchmarks)
# -------------------------------------------------------------
st.markdown('<div class="main-title">EE136IU • Topic 1: Comprehensive Model Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">📡 Wireless Intelligence & Automatic Modulation Classification (Tier 2: 3-Layer MLP)</div>', unsafe_allow_html=True)

tab_live, tab_bench, tab_arch = st.tabs(["⚡ Live Model Inference", "📊 Benchmark Performance Gallery", "🧠 Network Architecture & Math"])

with tab_live:
    # Inference
    x_raw = current_sample[feature_cols].values.reshape(1, -1)
    x_scaled = scaler.transform(x_raw)
    
    t_start = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_scaled.astype(np.float32)))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    t_end = time.perf_counter()
    latency_us = (t_end - t_start) * 1e6
    
    pred_idx = np.argmax(probs)
    pred_label = idx_to_class[pred_idx]
    confidence = probs[pred_idx] * 100
    is_correct = (pred_label == true_label)
    
    # Header Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #64748B;">MỨC SNR KIỂM THỬ</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #0284C7;">{selected_snr} dB</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #64748B;">LOẠI SÓNG THỰC TẾ (GROUND TRUTH)</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #334155;">{true_label}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        status_html = f'<span class="success-badge">✅ CHÍNH XÁC</span>' if is_correct else f'<span class="error-badge">❌ NHẦM LẪN</span>'
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #64748B;">DỰ ĐOÁN CỦA MLP</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {'#16A34A' if is_correct else '#DC2626'};">{pred_label}</div>
            {status_html}
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #64748B;">TỐC ĐỘ SUY LUẬN (LATENCY)</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #8B5CF6;">{latency_us:.2f} µs</div>
            <div style="font-size: 0.8rem; color: #10B981; font-weight: 600;">Thời gian thực (Real-time)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Biểu đồ phân phối xác suất 8 lớp
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("🎯 Phân phối xác suất 8 loại điều chế (Softmax Confidence)")
        
        df_probs = pd.DataFrame({
            'Modulation': classes,
            'Probability (%)': probs * 100
        }).sort_values(by='Probability (%)', ascending=True)
        
        colors = []
        for mod in df_probs['Modulation']:
            if mod == true_label and mod == pred_label:
                colors.append('#10B981') # Đúng: Xanh lá
            elif mod == pred_label and not is_correct:
                colors.append('#EF4444') # Đoán sai: Đỏ
            elif mod == true_label and not is_correct:
                colors.append('#F59E0B') # Thực tế bị bỏ lỡ: Vàng cam
            else:
                colors.append('#94A3B8') # Còn lại: Xám
                
        fig_prob, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        bars = ax.barh(df_probs['Modulation'], df_probs['Probability (%)'], color=colors, height=0.6)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Độ tin cậy xác suất (%)", fontsize=10, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                    va='center', ha='left', fontsize=9, fontweight='bold', color='#334155')
            
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close()
        
    with col_right:
        st.subheader("🔬 18 Đặc trưng vật lý của mẫu này (Radar/Bar Profile)")
        feat_vals = x_scaled[0]
        
        df_feat = pd.DataFrame({
            'Feature': feature_cols,
            'Normalized Value': feat_vals
        })
        
        fig_feat, ax_f = plt.subplots(figsize=(7, 4.5), dpi=150)
        ax_f.bar(df_feat['Feature'], df_feat['Normalized Value'], color='#0284C7', width=0.5)
        ax_f.set_xticklabels(feature_cols, rotation=70, ha='right', fontsize=8)
        ax_f.set_ylabel("Standardized Value (Z-Score)", fontsize=9, fontweight='bold')
        ax_f.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_feat)
        plt.close()

with tab_bench:
    st.subheader("📈 Báo cáo thực nghiệm toàn diện trên 15 mức SNR")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("**Đường cong Accuracy & Macro-F1 theo SNR (-10 dB đến +18 dB):**")
        p1 = os.path.join(BASE_DIR, "mlp_accuracy_vs_snr_large.png")
        if os.path.exists(p1):
            st.image(p1, use_container_width=True)
        else:
            st.info("Đồ thị Accuracy vs SNR đã được lưu trong Slide Báo Cáo")
    with b_col2:
        st.markdown("**Đường cong Macro-F1 Score theo SNR:**")
        p2 = os.path.join(BASE_DIR, "mlp_macro_f1_vs_snr_large.png")
        if os.path.exists(p2):
            st.image(p2, use_container_width=True)
        else:
            st.info("Đồ thị Macro-F1 đã được lưu trong Slide Báo Cáo")
        
    st.markdown("---")
    st.subheader("🎯 Ma trận nhầm lẫn (Confusion Matrices) tại các mốc then chốt:")
    cm_c1, cm_c2, cm_c3 = st.columns(3)
    with cm_c1:
        p_cm1 = os.path.join(BASE_DIR, "mlp_cm_low_snr_minus4dB.png")
        if os.path.exists(p_cm1):
            st.image(p_cm1, caption="Low SNR (-4 dB) - Nhiễu nặng", use_container_width=True)
        else:
            st.caption("Low SNR (-4 dB): 16QAM/64QAM/256QAM bị xoắn cụm do nhiễu.")
    with cm_c2:
        p_cm2 = os.path.join(BASE_DIR, "mlp_cm_mid_snr_plus4dB.png")
        if os.path.exists(p_cm2):
            st.image(p_cm2, caption="Mid SNR (+4 dB) - Chuyển tiếp", use_container_width=True)
        else:
            st.caption("Mid SNR (+4 dB): Tách hoàn toàn BPSK và QPSK.")
    with cm_c3:
        p_cm3 = os.path.join(BASE_DIR, "mlp_cm_high_snr_plus14dB.png")
        if os.path.exists(p_cm3):
            st.image(p_cm3, caption="High SNR (+14 dB) - Sóng khỏe", use_container_width=True)
        else:
            st.caption("High SNR (+14 dB): Tách rời các chòm sao PSK bậc cao.")
        
    st.markdown("---")
    st.subheader("📋 Bảng thông số đo đạc kỹ thuật (Executive Summary):")
    df_exec = pd.read_csv(os.path.join(BASE_DIR, "mlp_executive_summary.csv"))
    st.dataframe(df_exec, use_container_width=True)

with tab_arch:
    st.subheader("🧠 Cấu trúc toán học & Thiết kế mạng 3-Layer MLP")
    st.markdown("""
    * **Kiến trúc mạng:**
      $$\\text{Input } (x \\in \\mathbb{R}^{18}) \\xrightarrow{W_1} \\text{Linear}(128) \\rightarrow \\text{BatchNorm} \\rightarrow \\text{ReLU} \\xrightarrow{W_2} \\text{Linear}(64) \\rightarrow \\text{BatchNorm} \\rightarrow \\text{ReLU} \\xrightarrow{W_3} \\text{Linear}(8)$$
    * **Hàm mất mát (Objective Loss):**
      $$J(\\theta) = -\\sum_{c=1}^{8} y_c \\ln(p_c) + \\frac{1}{2}\\lambda \\|W\\|_2^2$$
    * **Bộ tối ưu (Optimizer):** `Adam` (Learning rate $\\eta = 10^{-3}$, weight decay $\\lambda = 10^{-4}$).
    * **Tại sao MLP vượt trội hơn SVM trong bài toán này?**
      * Thời gian suy luận chỉ **1.54 µs** (nhanh hơn SVM đa lớp với kernel RBF).
      * Tầng ẩn thứ 2 học được tương tác phi tuyến bậc cao giữa các cumulants ($C_{40}, C_{42}$) hiệu quả hơn các hàm nhân cổ điển.
    """)
    p_loss = os.path.join(BASE_DIR, "mlp_loss_history.png")
    if os.path.exists(p_loss):
        st.image(p_loss, caption="Quá trình hội tụ của hàm mất mát qua 35 Epochs", width=700)
    else:
        st.info("Mô hình hội tụ tại Epoch 33 với validation loss nhỏ nhất.")

