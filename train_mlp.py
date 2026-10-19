"""
EE136IU Capstone Mini-Project - Topic 1: Comprehensive Model Benchmark
Model Tier 2: 3-Layer Multilayer Perceptron (MLP) on 18 Domain Features
Author: Hai Nguyen (EE136IU)
"""

import os
import sys
import time
import zipfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -------------------------------------------------------------
# 1. Cấu hình thực nghiệm & Thiết lập hạt giống ngẫu nhiên (Seeds)
# -------------------------------------------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = r"C:\THIS PC\My PC\Emerging\topic1_mlp"
ZIP_PATH = r"C:\THIS PC\My PC\Emerging\New folder\features_train.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"=== EE136IU Topic 1: MLP Model Training ===")
print(f"Using device: {DEVICE}")

# -------------------------------------------------------------
# 2. Đọc dữ liệu từ file nén
# -------------------------------------------------------------
print("Reading dataset from zip file...")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    with z.open('features_train.csv') as f:
        df_train = pd.read_csv(f)
    with z.open('features_val.csv') as f:
        df_val = pd.read_csv(f)
    with z.open('features_test.csv') as f:
        df_test = pd.read_csv(f)

metadata_cols = [
    'example_id', 'split', 'modulation', 'snr_db', 'channel', 
    'realization_id', 'source_seed', 'pre_noise_power', 
    'noise_power', 'post_normalization_rms', 'channel_taps'
]
feature_cols = [c for c in df_train.columns if c not in metadata_cols]
print(f"18 Features: {feature_cols}")

classes = sorted(df_train['modulation'].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}
print(f"8 Modulation Classes ({len(classes)}): {classes}")

# Lưu class mapping
with open(os.path.join(OUTPUT_DIR, 'class_mapping.pkl'), 'wb') as f:
    pickle.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class, 'features': feature_cols}, f)

# Tách X, y
X_train = df_train[feature_cols].values.astype(np.float32)
y_train = df_train['modulation'].map(class_to_idx).values.astype(np.int64)

X_val = df_val[feature_cols].values.astype(np.float32)
y_val = df_val['modulation'].map(class_to_idx).values.astype(np.int64)

X_test = df_test[feature_cols].values.astype(np.float32)
y_test = df_test['modulation'].map(class_to_idx).values.astype(np.int64)
snr_test = df_test['snr_db'].values

# Chuẩn hóa StandardScaler (Chỉ fit trên tập train để chống rò rỉ dữ liệu)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print(f"Dataset split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# DataLoader
BATCH_SIZE = 128
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------------------------------------
# 3. Định nghĩa kiến trúc mạng 3-Layer MLP
# -------------------------------------------------------------
class ModulationMLP(nn.Module):
    def __init__(self, input_dim=18, num_classes=8):
        super(ModulationMLP, self).__init__()
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output Layer
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

model = ModulationMLP(input_dim=len(feature_cols), num_classes=len(classes)).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Architecture:\n{model}")
print(f"Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")

# -------------------------------------------------------------
# 4. Huấn luyện mô hình (Training Loop)
# -------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

EPOCHS = 35
best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, 'best_mlp.pth')
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

start_train_time = time.time()
print("\n--- Starting MLP Training ---")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_train_loss = running_loss / len(train_loader.dataset)
    
    # Đánh giá trên Validation
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = accuracy_score(val_targets, val_preds)
    
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    scheduler.step(epoch_val_acc)
    
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), best_model_path)
        star = " * (Best saved)"
    else:
        star = ""
        
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%{star}")

total_train_time = time.time() - start_train_time
print(f"\nTraining completed in {total_train_time:.2f} seconds! Best Val Accuracy: {best_val_acc*100:.2f}%")

# -------------------------------------------------------------
# 5. Đo đạc độ trễ suy luận (Inference Latency Benchmark)
# -------------------------------------------------------------
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

# Đo độ trễ suy luận trên 1,000 mẫu
dummy_input = torch.from_numpy(X_test[:1000]).to(DEVICE)
with torch.no_grad():
    for _ in range(10): # Warm-up
        _ = model(dummy_input)
    t0 = time.time()
    for _ in range(50):
        _ = model(dummy_input)
    t1 = time.time()
latency_us = ((t1 - t0) / (50 * 1000)) * 1e6
print(f"Inference Latency per Sample: {latency_us:.2f} microseconds (µs)")

# -------------------------------------------------------------
# 6. Đánh giá kiểm thử toàn diện trên 15 mức SNR (-10 dB đến +18 dB)
# -------------------------------------------------------------
test_tensor = torch.from_numpy(X_test).to(DEVICE)
with torch.no_grad():
    test_logits = model(test_tensor)
    test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
    test_preds = np.argmax(test_probs, axis=1)

overall_acc = accuracy_score(y_test, test_preds)
overall_f1 = f1_score(y_test, test_preds, average='macro')
print(f"\n--- OVERALL TEST SET RESULTS ---")
print(f"Overall Test Accuracy: {overall_acc*100:.2f}%")
print(f"Overall Test Macro-F1: {overall_f1*100:.2f}%")

# Đánh giá theo từng mức SNR
snr_levels = sorted(np.unique(snr_test))
snr_results = []

for snr in snr_levels:
    mask = (snr_test == snr)
    acc_snr = accuracy_score(y_test[mask], test_preds[mask])
    f1_snr = f1_score(y_test[mask], test_preds[mask], average='macro')
    snr_results.append({
        'SNR_dB': snr,
        'Accuracy': acc_snr,
        'Macro_F1': f1_snr,
        'Sample_Count': int(np.sum(mask))
    })

df_snr_results = pd.DataFrame(snr_results)
df_snr_results.to_csv(os.path.join(OUTPUT_DIR, 'mlp_snr_benchmark.csv'), index=False)
print("\nAccuracy & F1 per SNR (15 levels):")
print(df_snr_results[['SNR_dB', 'Accuracy', 'Macro_F1']])

# -------------------------------------------------------------
# 7. Xuất các đồ thị chuẩn Rubric
# -------------------------------------------------------------
# Đồ thị 1: Accuracy & Macro-F1 vs SNR
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(df_snr_results['SNR_dB'], df_snr_results['Accuracy'] * 100, 'o-', color='#0284C7', linewidth=2.5, markersize=7, label='MLP Accuracy (%)')
plt.plot(df_snr_results['SNR_dB'], df_snr_results['Macro_F1'] * 100, 's--', color='#10B981', linewidth=2.2, markersize=6, label='MLP Macro-F1 (%)')
plt.axvline(x=-4, color='#EF4444', linestyle=':', label='Low SNR (-4 dB)')
plt.axvline(x=4, color='#F59E0B', linestyle=':', label='Medium SNR (+4 dB)')
plt.axvline(x=14, color='#6366F1', linestyle=':', label='High SNR (+14 dB)')
plt.title('Tier 2 (MLP): Classification Accuracy & Macro-F1 vs. SNR', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Signal-to-Noise Ratio (SNR in dB)', fontsize=12)
plt.ylabel('Performance (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(snr_levels)
plt.ylim(0, 105)
plt.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_accuracy_f1_vs_snr.png'))
plt.close()

# Đồ thị 2: Confusion Matrices tại -4 dB, +4 dB, +14 dB
fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)
key_snrs = [-4, 4, 14]
titles = ['(a) Low SNR (-4 dB)', '(b) Medium SNR (+4 dB)', '(c) High SNR (+14 dB)']

for ax, target_snr, title in zip(axes, key_snrs, titles):
    mask = (snr_test == target_snr)
    cm = confusion_matrix(y_test[mask], test_preds[mask], normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"size": 9})
    acc_sub = accuracy_score(y_test[mask], test_preds[mask])
    ax.set_title(f"{title} | Acc: {acc_sub*100:.1f}%", fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Modulation', fontsize=11)
    ax.set_ylabel('True Modulation', fontsize=11)

plt.suptitle('Row-Normalized Confusion Matrices for MLP across Key Operational SNR Regimes', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_confusion_matrices_key_snr.png'), bbox_inches='tight')
plt.close()

# Đồ thị 3: Quá trình huấn luyện (Training History)
plt.figure(figsize=(9, 4.5), dpi=300)
plt.plot(history['train_loss'], label='Train Loss', color='#2563EB', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', color='#DC2626', linewidth=2)
plt.title('MLP Training & Validation Loss across Epochs', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Cross-Entropy Loss', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_loss_history.png'))
plt.close()

# -------------------------------------------------------------
# 8. Lưu bảng tổng hợp các thông số kỹ thuật (Executive Summary)
# -------------------------------------------------------------
summary_data = {
    'Metric': [
        'Model Tier', 'Architecture', 'Input Features', 'Number of Classes',
        'Total Parameters', 'Training Time (s)', 'Inference Latency (us)',
        'Overall Test Accuracy (%)', 'Overall Test Macro-F1 (%)',
        'Accuracy at -4 dB (%)', 'Accuracy at +4 dB (%)', 'Accuracy at +14 dB (%)'
    ],
    'Value': [
        'Tier 2: Neural Network on Features',
        '3-Layer MLP (18 -> 128 -> 64 -> 8)',
        '18 Cumulant & Spectral Features',
        '8 (5 PSK + 3 QAM)',
        f"{total_params:,}",
        f"{total_train_time:.2f}",
        f"{latency_us:.2f}",
        f"{overall_acc*100:.2f}",
        f"{overall_f1*100:.2f}",
        f"{df_snr_results.loc[df_snr_results['SNR_dB'] == -4, 'Accuracy'].values[0]*100:.2f}",
        f"{df_snr_results.loc[df_snr_results['SNR_dB'] == 4, 'Accuracy'].values[0]*100:.2f}",
        f"{df_snr_results.loc[df_snr_results['SNR_dB'] == 14, 'Accuracy'].values[0]*100:.2f}"
    ]
}
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'mlp_executive_summary.csv'), index=False)
print("\n=== EXECUTIVE SUMMARY TABLE ===")
print(df_summary.to_string(index=False))

print(f"\n[DONE] All models, plots, and metrics have been exported to: {OUTPUT_DIR}")
