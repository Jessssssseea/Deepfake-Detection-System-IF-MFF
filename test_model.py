# -*- coding: utf-8 -*-
"""
python test_model.py --real_dir ./real --fake_dir ./fake \
                     --model isolation_forest_model.pkl \
                     --scaler scaler.pkl \
                     --threshold threshold.txt \
                     --predictor shape_predictor_68_face_landmarks.dat \
                     --output_dir ./results
"""

import os
import sys
import argparse
import pickle
import glob
import cv2
import dlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2, fftshift
from tqdm import tqdm

# ====================== 配置 ======================
DEFAULT_MODEL_PATH = "isolation_forest_model_202602232010.pkl"
DEFAULT_SCALER_PATH = "scaler_202602232010.pkl"
DEFAULT_THRESHOLD_PATH = "threshold_202602232010.txt"
DEFAULT_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# ==================================================

# 初始化dlib
detector = dlib.get_frontal_face_detector()
predictor = None

def extract_features(image_path, predictor):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    face = faces[0]

    landmarks = predictor(gray, face)
    points = np.array([(p.x, p.y) for p in landmarks.parts()])

    # 1. 几何特征（对称性、比例）
    sym_pairs = [(36,45), (39,42), (48,54), (31,35)]
    sym_diff = []
    for i, j in sym_pairs:
        dist_left = np.linalg.norm(points[i] - points[27])
        dist_right = np.linalg.norm(points[j] - points[27])
        sym_diff.append(abs(dist_left - dist_right))
    face_width = face.right() - face.left()
    eye_width = np.linalg.norm(points[45] - points[36])
    mouth_width = np.linalg.norm(points[54] - points[48])
    ratios = [eye_width/face_width, mouth_width/face_width]
    geo_features = np.array(sym_diff + ratios).flatten()

    # 2. 光照一致性
    left_face = img[face.top():face.bottom(), face.left():(face.left()+face.right())//2]
    right_face = img[face.top():face.bottom(), (face.left()+face.right())//2:face.right()]
    left_mean = np.mean(cv2.cvtColor(left_face, cv2.COLOR_BGR2GRAY)) if left_face.size else 0
    right_mean = np.mean(cv2.cvtColor(right_face, cv2.COLOR_BGR2GRAY)) if right_face.size else 0
    illum_diff = abs(left_mean - right_mean)

    # 3. 纹理特征
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    high_freq_energy = np.sum(np.abs(laplacian)) / (gray.shape[0]*gray.shape[1])

    # 4. 颜色分布
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    skin_hist = cv2.calcHist([ycrcb], [1], skin_mask, [32], [0,256]).flatten()
    skin_hist = skin_hist / (skin_hist.sum() + 1e-6)
    mouth_pts = points[48:61]
    mouth_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mouth_mask, [mouth_pts], 255)
    mouth_mean = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[mouth_mask==255], axis=0) if np.any(mouth_mask) else np.zeros(3)
    cheek_rect = [face.left()+face.width()//4, face.top()+face.height()//2, face.width()//2, face.height()//4]
    cheek_roi = img[cheek_rect[1]:cheek_rect[1]+cheek_rect[3], cheek_rect[0]:cheek_rect[0]+cheek_rect[2]]
    cheek_mean = np.mean(cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2RGB), axis=(0,1)) if cheek_roi.size else np.zeros(3)
    lip_skin_diff = np.linalg.norm(mouth_mean - cheek_mean)

    # 5. 频域特征
    gray_resized = cv2.resize(gray, (64,64))
    fft = fft2(gray_resized)
    fft_shift = fftshift(fft)
    magnitude = np.abs(fft_shift)
    center = magnitude.shape[0]//2
    low_freq = magnitude[center-5:center+5, center-5:center+5].sum()
    total_energy = magnitude.sum()
    low_freq_ratio = low_freq / (total_energy + 1e-6)
    high_freq_ratio = 1 - low_freq_ratio

    # 6. 眼球反射
    left_eye_pts = points[36:42]
    right_eye_pts = points[42:48]
    def extract_eye_region(pts):
        x, y, w, h = cv2.boundingRect(pts)
        eye_img = img[y:y+h, x:x+w]
        if eye_img.size == 0:
            return None
        return cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    left_eye_img = extract_eye_region(left_eye_pts)
    right_eye_img = extract_eye_region(right_eye_pts)
    if left_eye_img is not None and right_eye_img is not None:
        left_high = np.sum(left_eye_img > 200) / left_eye_img.size
        right_high = np.sum(right_eye_img > 200) / right_eye_img.size
        eye_high_diff = abs(left_high - right_high)
    else:
        eye_high_diff = 0

    # 组合所有特征
    features = np.concatenate([
        geo_features,
        [illum_diff],
        hist_lbp[:30],
        [high_freq_energy],
        skin_hist[:10],
        [lip_skin_diff],
        [low_freq_ratio, high_freq_ratio],
        [eye_high_diff]
    ])
    return features

def process_directory(dir_path, label, model, scaler, threshold, predictor, ext_list=('*.jpg','*.jpeg','*.png')):
    features_list = []
    scores_list = []
    preds_list = []
    true_labels = []
    img_paths_success = []
    fail_count = 0

    image_paths = []
    for ext in ext_list:
        image_paths.extend(glob.glob(os.path.join(dir_path, ext)))
        image_paths.extend(glob.glob(os.path.join(dir_path, ext.upper())))

    if not image_paths:
        print(f"Warning: No images found in directory {dir_path}")
        return [], [], [], [], [], 0

    for img_path in tqdm(image_paths, desc=f"Processing {os.path.basename(dir_path)}"):
        feat = extract_features(img_path, predictor)
        if feat is None:
            fail_count += 1
            continue
        feat_scaled = scaler.transform([feat])
        score = -model.decision_function(feat_scaled)[0]
        pred = 1 if score > threshold else 0   # 1表示Fake, 0表示Real

        features_list.append(feat)
        scores_list.append(score)
        preds_list.append(pred)
        true_labels.append(label)  # 1表示Fake, 0表示Real
        img_paths_success.append(img_path)

    return features_list, scores_list, preds_list, true_labels, img_paths_success, fail_count

def save_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#E74C3C', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.15, color='#E74C3C')
    plt.plot([0, 1], [0, 1], color='#34495E', lw=2, linestyle='--', alpha=0.7)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return roc_auc

def save_pr_curve(y_true, y_score, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#3498DB', lw=2.5, label=f'PR curve (AP = {ap:.4f})')
    plt.fill_between(recall, precision, alpha=0.15, color='#3498DB')

    # 基准线（随机分类器）
    pos_ratio = np.sum(y_true) / len(y_true)
    plt.axhline(y=pos_ratio, color='#34495E', linestyle='--', alpha=0.7,
                label=f'Baseline ({pos_ratio:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return ap

def save_confusion_matrix(y_true, y_pred, save_path, labels=['Real', 'Fake']):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    values = [[tn, fp], [fn, tp]]
    max_val = max(tn, fp, fn, tp)
    if max_val == 0:
        return
      
    fig, ax = plt.subplots(figsize=(6, 5))
    scale = 0.8

    for i in range(2):          # 真实标签 i=0:Real, i=1:Fake
        for j in range(2):      # 预测标签 j=0:Real, j=1:Fake
            val = values[i][j]
            if val == 0:
                continue
            # 矩形的宽度和高度与 val 的平方根成正比，使得面积与 val 成正比
            size = np.sqrt(val / max_val) * scale
            w = size
            h = size
            center_x = j
            center_y = 1 - i
            rect = plt.Rectangle(
                (center_x - w/2, center_y - h/2), w, h,
                facecolor=plt.cm.Blues(val / max_val),
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            text_color = 'white' if val/max_val > 0.5 else 'black'
            ax.text(center_x, center_y, str(val),
                    ha='center', va='center', fontsize=12, color=text_color)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels[::-1])  # 反转使顶部为Real，底部为Fake
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title('Confusion Matrix (area ∝ count)', fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_score_distribution(y_true, y_score, save_path, threshold_val=None):
    real_scores = [y_score[i] for i in range(len(y_true)) if y_true[i] == 0]
    fake_scores = [y_score[i] for i in range(len(y_true)) if y_true[i] == 1]

    plt.figure(figsize=(8, 6))
    plt.hist(real_scores, bins=30, alpha=0.7, label='Real', color='#2ECC71', density=True, edgecolor='white', linewidth=0.5)
    plt.hist(fake_scores, bins=30, alpha=0.7, label='Fake', color='#E74C3C', density=True, edgecolor='white', linewidth=0.5)

    if threshold_val is not None:
        plt.axvline(x=threshold_val, color='#34495E', linestyle='--', linewidth=2, label=f'Threshold = {threshold_val:.6f}')

    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Anomaly Score Distribution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_pca_scatter(features, y_true, save_path):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))

    real_mask = np.array(y_true) == 0
    fake_mask = np.array(y_true) == 1

    plt.scatter(features_2d[real_mask, 0], features_2d[real_mask, 1],
                c='#2ECC71', alpha=0.6, edgecolors='k', linewidth=0.5,
                label='Real', s=50)
    plt.scatter(features_2d[fake_mask, 0], features_2d[fake_mask, 1],
                c='#E74C3C', alpha=0.6, edgecolors='k', linewidth=0.5,
                label='Fake', s=50)

    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.title('PCA Visualization of Feature Space', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='AI Face Forgery Model Testing Tool')
    parser.add_argument('--real_dir', type=str, required=True, help='Directory containing real face images')
    parser.add_argument('--fake_dir', type=str, required=True, help='Directory containing fake face images')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Trained Isolation Forest model file (.pkl)')
    parser.add_argument('--scaler', type=str, default=DEFAULT_SCALER_PATH, help='Scaler file (.pkl)')
    parser.add_argument('--threshold', type=str, default=DEFAULT_THRESHOLD_PATH, help='Threshold file (.txt)')
    parser.add_argument('--predictor', type=str, default=DEFAULT_PREDICTOR_PATH, help='dlib landmark predictor file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for charts and report (default: model_name + " test results")')
    args = parser.parse_args()

    # 确定输出目录名称：模型名称 + " test results"
    if args.output_dir is None:
        model_basename = os.path.splitext(os.path.basename(args.model))[0]
        args.output_dir = model_basename + " test results"
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型、标准化器、阈值
    print("Loading model...")
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist")
        sys.exit(1)
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    if not os.path.exists(args.scaler):
        print(f"Error: Scaler file {args.scaler} does not exist")
        sys.exit(1)
    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)

    if not os.path.exists(args.threshold):
        print(f"Error: Threshold file {args.threshold} does not exist")
        sys.exit(1)
    with open(args.threshold, 'r') as f:
        threshold = float(f.read().strip())

    if not os.path.exists(args.predictor):
        print(f"Error: dlib predictor {args.predictor} does not exist")
        sys.exit(1)
    global predictor
    predictor = dlib.shape_predictor(args.predictor)

    print(f"Model loaded successfully. Threshold = {threshold:.6f}")

    # 处理真实图片 (标签0)
    real_feat, real_scores, real_preds, real_true, real_paths, real_fail = process_directory(
        args.real_dir, 0, model, scaler, threshold, predictor
    )
    # 处理伪造图片 (标签1)
    fake_feat, fake_scores, fake_preds, fake_true, fake_paths, fake_fail = process_directory(
        args.fake_dir, 1, model, scaler, threshold, predictor
    )

    # 合并数据
    all_features = np.array(real_feat + fake_feat)
    all_scores = np.array(real_scores + fake_scores)
    all_preds = np.array(real_preds + fake_preds)
    all_true = np.array(real_true + fake_true)

    # 统计
    total_real = len(real_paths) + real_fail
    total_fake = len(fake_paths) + fake_fail
    success_real = len(real_paths)
    success_fake = len(fake_paths)

    print("\n========== Test Statistics ==========")
    print(f"Real faces: Total {total_real}, Detected {success_real}, Failed {real_fail}")
    print(f"Fake faces: Total {total_fake}, Detected {success_fake}, Failed {fake_fail}")
    if success_real + success_fake == 0:
        print("No successfully detected faces, cannot evaluate performance.")
        sys.exit(0)

    # 计算指标
    print("\n========== Classification Metrics ==========")
    cm = confusion_matrix(all_true, all_preds)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Fake accuracy):    {recall:.4f}")
    print(f"Specificity (Real accuracy): {specificity:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=['Real', 'Fake']))

    # 计算AUC和AP
    if len(np.unique(all_true)) == 2:
        roc_auc = auc(*roc_curve(all_true, all_scores)[:2])
        ap = average_precision_score(all_true, all_scores)
        print(f"AUC: {roc_auc:.4f}")
        print(f"AP:  {ap:.4f}")
    else:
        roc_auc, ap = None, None
        print("Warning: Only one class present, cannot compute AUC/AP")

    # 保存图表
    print(f"\nGenerating charts and saving to {args.output_dir} ...")
    if roc_auc is not None:
        save_roc_curve(all_true, all_scores, os.path.join(args.output_dir, 'roc_curve.png'))
    if ap is not None:
        save_pr_curve(all_true, all_scores, os.path.join(args.output_dir, 'pr_curve.png'))
    save_confusion_matrix(all_true, all_preds, os.path.join(args.output_dir, 'confusion_matrix.png'))
    save_score_distribution(all_true, all_scores, os.path.join(args.output_dir, 'score_distribution.png'), threshold_val=threshold)
    if len(all_features) > 1:
        save_pca_scatter(all_features, all_true, os.path.join(args.output_dir, 'pca_scatter.png'))

    report_path = os.path.join(args.output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========== AI Face Forgery Model Test Report ==========\n")
        f.write(f"Model file: {args.model}\n")
        f.write(f"Threshold: {threshold:.6f}\n\n")
        f.write(f"Real images total: {total_real}, successful: {success_real}, failed: {real_fail}\n")
        f.write(f"Fake images total: {total_fake}, successful: {success_fake}, failed: {fake_fail}\n\n")
        f.write("Classification Metrics:\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall (Fake accuracy):    {recall:.4f}\n")
        f.write(f"Specificity (Real accuracy): {specificity:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        if roc_auc is not None:
            f.write(f"AUC: {roc_auc:.4f}\n")
        if ap is not None:
            f.write(f"AP:  {ap:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(all_true, all_preds, target_names=['Real', 'Fake']))

    print(f"Report saved to: {report_path}")
    print("All charts generated successfully!")

if __name__ == '__main__':
    main()
