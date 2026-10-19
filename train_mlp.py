"""
EE136IU Capstone Mini-Project - Topic 1: Comprehensive Model Benchmark
Model Tier 2: 3-Layer Multilayer Perceptron (MLP) on 18 Domain Features
Author: Hai Nguyen (EEEEIU22093)
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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model import ModulationMLP, MLP, load_pretrained_mlp

__all__ = ["ModulationMLP", "MLP", "load_pretrained_mlp", "main"]


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # -------------------------------------------------------------
    # 1. Cấu hình thực nghiệm & Thiết lập hạt giống ngẫu nhiên (Seeds)
    # -------------------------------------------------------------
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Tự động tìm file nén dữ liệu
    candidate_zips = [
        os.path.join(OUTPUT_DIR, "features_train.zip"),
        os.path.join(os.path.dirname(OUTPUT_DIR), "New folder", "features_train.zip"),
        r"C:\THIS PC\My PC\Emerging\New folder\features_train.zip",
    ]
    ZIP_PATH = next((p for p in candidate_zips if os.path.exists(p)), candidate_zips[0])

    print("=== EE136IU Topic 1: MLP Model Training ===")
    print(f"Using device: {DEVICE}")

    # -------------------------------------------------------------
    # 2. Đọc dữ liệu từ file nén
    # -------------------------------------------------------------
    print(f"Reading dataset from: {ZIP_PATH}")
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

    with open(os.path.join(OUTPUT_DIR, 'class_mapping.pkl'), 'wb') as f:
        pickle.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class, 'features': feature_cols}, f)

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

    BATCH_SIZE = 128
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------------------------------------------
    # 3. Khởi tạo kiến trúc mạng 3-Layer MLP
    # -------------------------------------------------------------
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
    # 5. Đo đạc độ trễ suy luận (Batched vs Single-Sample Latency)
    # -------------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # (a) Amortized Batched Latency (B = 1,000 bursts)
    dummy_batch = torch.from_numpy(X_test[:1000]).to(DEVICE)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_batch)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(dummy_batch)
        t1 = time.perf_counter()
    batched_latency_us = ((t1 - t0) / (50 * 1000)) * 1e6

    # (b) Single-Burst Dispatch Latency (B = 1 burst)
    dummy_single = torch.from_numpy(X_test[:1]).to(DEVICE)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_single)
        t0 = time.perf_counter()
        for _ in range(500):
            _ = model(dummy_single)
        t1 = time.perf_counter()
    single_latency_us = ((t1 - t0) / 500) * 1e6

    print(f"Batched Throughput Latency (B=1000): {batched_latency_us:.2f} µs/burst")
    print(f"Single-Sample Dispatch Latency (B=1): {single_latency_us:.2f} µs/burst")

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

    snr_levels = sorted(np.unique(snr_test))
    snr_results = []
    for snr in snr_levels:
        mask = (snr_test == snr)
        acc_snr = accuracy_score(y_test[mask], test_preds[mask])
        f1_snr = f1_score(y_test[mask], test_preds[mask], average='macro')
        snr_results.append({
            'SNR_dB': int(snr),
            'Accuracy': acc_snr * 100,
            'Macro_F1': f1_snr * 100,
            'Count': int(np.sum(mask))
        })

    df_snr_results = pd.DataFrame(snr_results)
    df_snr_results.to_csv(os.path.join(OUTPUT_DIR, 'mlp_snr_benchmark_verified.csv'), index=False)

    # Xuất bảng Executive Summary
    summary_data = {
        'Metric': [
            'Model Tier', 'Architecture', 'Input Features', 'Number of Classes',
            'Total Parameters', 'Training Time (s)',
            'Batched Throughput Latency (us/sample @ B=1000)',
            'Single-Sample Dispatch Latency (us/sample @ B=1)',
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
            f"{batched_latency_us:.2f}",
            f"~{single_latency_us:.1f}",
            f"{overall_acc*100:.2f}",
            f"{overall_f1*100:.2f}",
            f"{df_snr_results.loc[df_snr_results['SNR_dB'] == -4, 'Accuracy'].values[0]:.2f}",
            f"{df_snr_results.loc[df_snr_results['SNR_dB'] == 4, 'Accuracy'].values[0]:.2f}",
            f"{df_snr_results.loc[df_snr_results['SNR_dB'] == 14, 'Accuracy'].values[0]:.2f}"
        ]
    }
    pd.DataFrame(summary_data).to_csv(os.path.join(OUTPUT_DIR, 'mlp_executive_summary.csv'), index=False)


if __name__ == "__main__":
    main()
