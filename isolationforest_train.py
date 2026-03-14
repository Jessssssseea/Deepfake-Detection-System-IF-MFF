# -*- coding: utf-8 -*-

import os
import sys
import cv2
import dlib
import numpy as np
import pickle
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2, fftshift
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = r"D:\AI face"
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "real")
FAKE_DIR = os.path.join(PROJECT_ROOT, "data", "fake")
MODEL_SAVE_PATH = "isolation_forest_model.pkl"
SCALER_SAVE_PATH = "scaler.pkl"
THRESHOLD_FILE = "threshold.txt"
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "training_log.txt")
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

class TeedOutput:
    def __init__(self, original_stdout, log_file):
        self.original_stdout = original_stdout
        self.log_file = log_file

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

def extract_features(image_path):
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
    # 68点索引：左眼外角36，右眼外角45；左眼内角39，右眼内角42；嘴角左48，右54等
    sym_pairs = [(36,45), (39,42), (48,54), (31,35)]
    sym_diff = []
    for i, j in sym_pairs:
        dist_left = np.linalg.norm(points[i] - points[27])  # 鼻尖27
        dist_right = np.linalg.norm(points[j] - points[27])
        sym_diff.append(abs(dist_left - dist_right))
    # 人脸比例特征：眼宽/脸宽、嘴宽/脸宽
    face_width = face.right() - face.left()
    eye_width = np.linalg.norm(points[45] - points[36])
    mouth_width = np.linalg.norm(points[54] - points[48])
    ratios = [eye_width/face_width, mouth_width/face_width]

    geo_features = np.array(sym_diff + ratios).flatten()

    # 2. 光照一致性特征（左右脸平均亮度差异）
    # 获取左脸区域（索引0-16为轮廓，取左半部分粗略）
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

    # 4. 颜色分布特征
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    skin_hist = cv2.calcHist([ycrcb], [1], skin_mask, [32], [0,256]).flatten()
    skin_hist = skin_hist / (skin_hist.sum() + 1e-6)

    mouth_pts = points[48:61]  # 嘴部关键点
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

    # 6. 眼球反射一致性
    # 左右眼区域（索引36-41左眼，42-47右眼）
    left_eye_pts = points[36:42]
    right_eye_pts = points[42:48]
    left_eye_img = extract_eye_region(img, left_eye_pts)
    right_eye_img = extract_eye_region(img, right_eye_pts)

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

def extract_eye_region(img, eye_pts):
    x, y, w, h = cv2.boundingRect(eye_pts)
    eye_img = img[y:y+h, x:x+w]
    if eye_img.size == 0:
        return None
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    return gray_eye

def build_dataset(folder, label):
    features_list = []
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    print(f"正在处理 {folder}，共 {len(paths)} 张图片...")
    for i, path in enumerate(paths):
        if i % 500 == 0:
            print(f"  已处理 {i}/{len(paths)}")
        feat = extract_features(path)
        if feat is not None:
            features_list.append(feat)
    X = np.array(features_list)
    y = np.full(len(X), label)
    return X, y

if __name__ == '__main__':
    original_stdout = sys.stdout
    log_file = None
    try:
        log_file = open(LOG_FILE_PATH, 'a', encoding='utf-8')
        sys.stdout = TeedOutput(original_stdout, log_file)

        print("="*50)
        print("基于多维特征 + 孤立森林的AI换脸检测系统训练开始")
        print("="*50)

        print("开始提取真实图片特征...")
        X_real, y_real = build_dataset(REAL_DIR, 0)
        print(f"真实图片有效特征数: {len(X_real)}")

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
        print(f"训练集: {len(X_train)} 验证集: {len(X_val)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print("训练孤立森林...")
        iso_forest = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
        iso_forest.fit(X_train_scaled)

        val_scores = -iso_forest.decision_function(X_val_scaled)  # 得分越高越异常
        threshold = np.percentile(val_scores, 95)
        print(f"验证集95%分位数阈值: {threshold:.4f}")

        print("提取伪造图片特征...")
        X_fake, y_fake = build_dataset(FAKE_DIR, 1)
        print(f"伪造图片有效特征数: {len(X_fake)}")
        X_fake_scaled = scaler.transform(X_fake)

        real_scores = -iso_forest.decision_function(X_val_scaled)
        fake_scores = -iso_forest.decision_function(X_fake_scaled)

        y_true = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
        y_scores = np.concatenate([real_scores, fake_scores])

        auc = roc_auc_score(y_true, y_scores)
        print(f"测试集AUC: {auc:.4f}")

        preds = (y_scores > threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        print(f"阈值 {threshold:.4f} 下准确率: {acc:.4f}")

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Isolation Forest (AUC={auc:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png')
        plt.show()

        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(iso_forest, f)
        with open(SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        with open(THRESHOLD_FILE, 'w') as f:
            f.write(str(threshold))
        print(f"模型已保存至 {MODEL_SAVE_PATH}")
        print(f"阈值已保存至 {THRESHOLD_FILE}")

        print("="*50)
        print("训练完成！所有日志已保存至文件")
        print("="*50)

    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise e
    finally:
        if log_file:
            sys.stdout = original_stdout
            log_file.close()
            print(f"\n训练日志已保存到: {LOG_FILE_PATH}")
