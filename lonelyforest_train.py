import os
import sys  # 新增：导入sys模块用于重定向输出
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

# ====================== 配置路径 ======================
PROJECT_ROOT = r"D:\AI face"                # 项目根目录
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "real")   # 真实图片文件夹
FAKE_DIR = os.path.join(PROJECT_ROOT, "data", "fake")   # 伪造图片文件夹
MODEL_SAVE_PATH = "isolation_forest_model.pkl"          # 模型保存路径
SCALER_SAVE_PATH = "scaler.pkl"                         # 标准化器保存路径
THRESHOLD_FILE = "threshold.txt"                        # 阈值文件
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "training_log.txt")  # 新增：日志文件保存路径

# dlib 模型路径（需提前下载 shape_predictor_68_face_landmarks.dat）
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# ======================================================

# 新增：自定义输出类 - 同时输出到控制台和文件
class TeedOutput:
    def __init__(self, original_stdout, log_file):
        self.original_stdout = original_stdout  # 原始控制台输出
        self.log_file = log_file                # 日志文件句柄

    def write(self, message):
        # 写入控制台
        self.original_stdout.write(message)
        # 写入日志文件（确保中文不乱码）
        self.log_file.write(message)
        # 立即刷新缓冲区，避免内容积压
        self.log_file.flush()

    def flush(self):
        # 刷新输出缓冲区
        self.original_stdout.flush()
        self.log_file.flush()

# 初始化 dlib 检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

# -------------------- 特征提取函数 --------------------
def extract_features(image_path):
    """
    从单张图片提取特征向量
    返回: 特征列表 (numpy array) 或 None（如果人脸检测失败）
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    face = faces[0]  # 取最大人脸

    # 获取关键点
    landmarks = predictor(gray, face)
    points = np.array([(p.x, p.y) for p in landmarks.parts()])

    # 1. 几何特征（对称性、比例）
    # 计算左右对称点对的距离差
    # 68点索引：左眼外角36，右眼外角45；左眼内角39，右眼内角42；嘴角左48，右54等
    sym_pairs = [(36,45), (39,42), (48,54), (31,35)]  # 示例对称点对
    sym_diff = []
    for i, j in sym_pairs:
        dist_left = np.linalg.norm(points[i] - points[27])  # 鼻尖27
        dist_right = np.linalg.norm(points[j] - points[27])
        sym_diff.append(abs(dist_left - dist_right))
    # 人脸比例特征：眼宽/脸宽、嘴宽/脸宽
    face_width = face.right() - face.left()
    eye_width = np.linalg.norm(points[45] - points[36])  # 右外-左外
    mouth_width = np.linalg.norm(points[54] - points[48])
    ratios = [eye_width/face_width, mouth_width/face_width]

    # 关键点置信度（这里用检测分数近似，dlib未直接提供，用0填充）
    # 可考虑用landmark的分数，但dlib不输出，暂用0
    geo_features = np.array(sym_diff + ratios).flatten()

    # 2. 光照一致性特征（左右脸平均亮度差异）
    # 获取左脸区域（索引0-16为轮廓，取左半部分粗略）
    left_face = img[face.top():face.bottom(), face.left():(face.left()+face.right())//2]
    right_face = img[face.top():face.bottom(), (face.left()+face.right())//2:face.right()]
    left_mean = np.mean(cv2.cvtColor(left_face, cv2.COLOR_BGR2GRAY)) if left_face.size else 0
    right_mean = np.mean(cv2.cvtColor(right_face, cv2.COLOR_BGR2GRAY)) if right_face.size else 0
    illum_diff = abs(left_mean - right_mean)

    # 3. 纹理特征
    # LBP直方图
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    # 高频噪声能量
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    high_freq_energy = np.sum(np.abs(laplacian)) / (gray.shape[0]*gray.shape[1])

    # 4. 颜色分布特征
    # 肤色区域（简单用YCrCb的Cr分量直方图）
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))  # 简单肤色范围
    skin_hist = cv2.calcHist([ycrcb], [1], skin_mask, [32], [0,256]).flatten()
    skin_hist = skin_hist / (skin_hist.sum() + 1e-6)
    # 唇色与肤色差
    mouth_pts = points[48:61]  # 嘴部关键点
    mouth_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mouth_mask, [mouth_pts], 255)
    mouth_mean = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[mouth_mask==255], axis=0) if np.any(mouth_mask) else np.zeros(3)
    # 取脸颊区域（简单用框）代表肤色
    cheek_rect = [face.left()+face.width()//4, face.top()+face.height()//2, face.width()//2, face.height()//4]
    cheek_roi = img[cheek_rect[1]:cheek_rect[1]+cheek_rect[3], cheek_rect[0]:cheek_rect[0]+cheek_rect[2]]
    cheek_mean = np.mean(cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2RGB), axis=(0,1)) if cheek_roi.size else np.zeros(3)
    lip_skin_diff = np.linalg.norm(mouth_mean - cheek_mean)

    # 5. 频域特征
    gray_resized = cv2.resize(gray, (64,64))
    fft = fft2(gray_resized)
    fft_shift = fftshift(fft)
    magnitude = np.abs(fft_shift)
    # 低频能量（中心区域）
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
    # 计算高光区域（灰度值高的区域比例）
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
        hist_lbp[:30],  # 取前30维
        [high_freq_energy],
        skin_hist[:10],  # 取前10维
        [lip_skin_diff],
        [low_freq_ratio, high_freq_ratio],
        [eye_high_diff]
    ])
    return features

def extract_eye_region(img, eye_pts):
    """从图像中提取眼部区域（矩形）"""
    x, y, w, h = cv2.boundingRect(eye_pts)
    eye_img = img[y:y+h, x:x+w]
    if eye_img.size == 0:
        return None
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    return gray_eye

# -------------------- 数据集构建 --------------------
def build_dataset(folder, label):
    """
    遍历文件夹，提取特征，返回特征矩阵和标签
    label: 0=真实, 1=伪造
    """
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

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    # 新增：初始化日志输出（核心修改）
    original_stdout = sys.stdout  # 保存原始控制台输出对象
    log_file = None
    try:
        # 打开日志文件（utf-8编码避免中文乱码，'w'表示覆盖原有内容，'a'为追加）
        log_file = open(LOG_FILE_PATH, 'a', encoding='utf-8')
        # 重定向输出：同时输出到控制台和日志文件
        sys.stdout = TeedOutput(original_stdout, log_file)

        # 原有训练逻辑
        print("="*50)
        print("基于多维特征 + 孤立森林的AI换脸检测系统训练开始")
        print("="*50)

        # 1. 构建真实数据集（用于训练和验证）
        print("开始提取真实图片特征...")
        X_real, y_real = build_dataset(REAL_DIR, 0)
        print(f"真实图片有效特征数: {len(X_real)}")

        # 划分训练集和验证集（70%训练，30%验证）
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
        print(f"训练集: {len(X_train)} 验证集: {len(X_val)}")

        # 2. 特征标准化（孤立森林不需要标准化，但为了后续可扩展，仍做）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 3. 训练孤立森林
        print("训练孤立森林...")
        iso_forest = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
        iso_forest.fit(X_train_scaled)

        # 4. 验证集得分和阈值选择
        val_scores = -iso_forest.decision_function(X_val_scaled)  # 得分越高越异常
        # 使用95%分位数作为阈值
        threshold = np.percentile(val_scores, 95)
        print(f"验证集95%分位数阈值: {threshold:.4f}")

        # 5. 构建伪造数据集（测试）
        print("提取伪造图片特征...")
        X_fake, y_fake = build_dataset(FAKE_DIR, 1)
        print(f"伪造图片有效特征数: {len(X_fake)}")
        X_fake_scaled = scaler.transform(X_fake)

        # 6. 在测试集上评估
        real_scores = -iso_forest.decision_function(X_val_scaled)  # 验证集真实
        fake_scores = -iso_forest.decision_function(X_fake_scaled) # 伪造集

        y_true = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
        y_scores = np.concatenate([real_scores, fake_scores])

        auc = roc_auc_score(y_true, y_scores)
        print(f"测试集AUC: {auc:.4f}")

        # 根据阈值计算准确率
        preds = (y_scores > threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        print(f"阈值 {threshold:.4f} 下准确率: {acc:.4f}")

        # 绘制ROC曲线
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

        # 保存模型和阈值
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
        # 捕获异常并记录
        print(f"训练过程中出现错误: {str(e)}")
        raise e
    finally:
        # 新增：恢复原始输出并关闭日志文件（核心修改）
        if log_file:
            sys.stdout = original_stdout  # 恢复控制台输出
            log_file.close()  # 关闭日志文件
            print(f"\n训练日志已保存到: {LOG_FILE_PATH}")

# -------------------- 单张图片预测函数 --------------------
def predict_single_image(image_path, model, scaler, threshold):
    """
    预测单张图片
    返回 (prediction, score)  prediction为'Real'或'Fake'
    """
    feat = extract_features(image_path)
    if feat is None:
        return "No face detected", None
    feat_scaled = scaler.transform([feat])
    score = -model.decision_function(feat_scaled)[0]
    pred = 'Fake' if score > threshold else 'Real'
    return pred, score

