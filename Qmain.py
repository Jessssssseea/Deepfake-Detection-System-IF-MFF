import sys
import os
import pickle
import random
import numpy as np
import cv2
import dlib
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2, fftshift
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QFileDialog, QLineEdit,
    QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem, QProgressBar,
    QTextEdit, QMessageBox, QHeaderView, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon

DEFAULT_MODEL_PATH = "isolation_forest_model_202602232010.pkl"
DEFAULT_SCALER_PATH = "scaler_202602232010.pkl"
DEFAULT_THRESHOLD_PATH = "threshold_202602232010.txt"
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FAVICON_PATH = "favicon.ico"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)


def extract_features_from_image(img_bgr):
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    face = faces[0]

    landmarks = predictor(gray, face)
    points = np.array([(p.x, p.y) for p in landmarks.parts()])

    # 1. 几何特征
    sym_pairs = [(36, 45), (39, 42), (48, 54), (31, 35)]
    sym_diff = []
    for i, j in sym_pairs:
        dist_left = np.linalg.norm(points[i] - points[27])
        dist_right = np.linalg.norm(points[j] - points[27])
        sym_diff.append(abs(dist_left - dist_right))
    face_width = face.right() - face.left()
    eye_width = np.linalg.norm(points[45] - points[36])
    mouth_width = np.linalg.norm(points[54] - points[48])
    ratios = [eye_width / face_width, mouth_width / face_width]
    geo_features = np.array(sym_diff + ratios).flatten()

    # 2. 光照一致性
    left_face = img_bgr[face.top():face.bottom(), face.left():(face.left() + face.right()) // 2]
    right_face = img_bgr[face.top():face.bottom(), (face.left() + face.right()) // 2:face.right()]
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
    high_freq_energy = np.sum(np.abs(laplacian)) / (gray.shape[0] * gray.shape[1])

    # 4. 颜色分布
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_hist = cv2.calcHist([ycrcb], [1], skin_mask, [32], [0, 256]).flatten()
    skin_hist = skin_hist / (skin_hist.sum() + 1e-6)
    mouth_pts = points[48:61]
    mouth_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mouth_mask, [mouth_pts], 255)
    mouth_mean = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[mouth_mask == 255], axis=0) if np.any(
        mouth_mask) else np.zeros(3)
    cheek_rect = [face.left() + face.width() // 4, face.top() + face.height() // 2, face.width() // 2,
                  face.height() // 4]
    cheek_roi = img_bgr[cheek_rect[1]:cheek_rect[1] + cheek_rect[3], cheek_rect[0]:cheek_rect[0] + cheek_rect[2]]
    cheek_mean = np.mean(cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2RGB), axis=(0, 1)) if cheek_roi.size else np.zeros(3)
    lip_skin_diff = np.linalg.norm(mouth_mean - cheek_mean)

    # 5. 频域特征
    gray_resized = cv2.resize(gray, (64, 64))
    fft = fft2(gray_resized)
    fft_shift = fftshift(fft)
    magnitude = np.abs(fft_shift)
    center = magnitude.shape[0] // 2
    low_freq = magnitude[center - 5:center + 5, center - 5:center + 5].sum()
    total_energy = magnitude.sum()
    low_freq_ratio = low_freq / (total_energy + 1e-6)
    high_freq_ratio = 1 - low_freq_ratio

    # 6. 眼球反射
    left_eye_pts = points[36:42]
    right_eye_pts = points[42:48]

    def extract_eye_region(pts):
        x, y, w, h = cv2.boundingRect(pts)
        eye_img = img_bgr[y:y + h, x:x + w]
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


def extract_features(image_path):
    img = cv2.imread(image_path)
    return extract_features_from_image(img)


def predict_image(image_bgr, model, scaler, threshold):
    feat = extract_features_from_image(image_bgr)
    if feat is None:
        return None, None, "未检测到人脸"
    feat_scaled = scaler.transform([feat])
    score = -model.decision_function(feat_scaled)[0]
    pred = 'Fake' if score > threshold else 'Real'
    return pred, score, "成功"


class ImageDetectionThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, image_path, model, scaler, threshold):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def run(self):
        img = cv2.imread(self.image_path)
        if img is None:
            self.finished.emit((None, None, "无法读取图片"))
            return
        pred, score, msg = predict_image(img, self.model, self.scaler, self.threshold)
        self.finished.emit((pred, score, msg))


class VideoDetectionThread(QThread):
    progress = pyqtSignal(int)  # 当前处理的帧序号（1开始）
    frame_result = pyqtSignal(int, str, float)  # 帧索引，预测，得分
    finished = pyqtSignal(list, dict)  # 全部结果列表，统计信息

    def __init__(self, video_path, frame_indices, model, scaler, threshold):
        super().__init__()
        self.video_path = video_path
        self.frame_indices = frame_indices  # 要检测的帧索引列表
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        results = []
        stats = {'real': 0, 'fake': 0, 'no_face': 0}

        for i, idx in enumerate(self.frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            pred, score, msg = predict_image(frame, self.model, self.scaler, self.threshold)
            if pred is None:
                stats['no_face'] += 1
                pred_display = '无人脸'
                score_display = 0.0
            else:
                stats['real' if pred == 'Real' else 'fake'] += 1
                pred_display = pred
                score_display = score
            results.append((idx, pred_display, score_display))
            self.frame_result.emit(idx, pred_display, score_display)
            self.progress.emit(i + 1)  # 进度

        cap.release()
        self.finished.emit(results, stats)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI换脸检测工具")
        self.resize(900, 600)

        self.set_window_icon()

        self.model = None
        self.scaler = None
        self.threshold = None

        self.init_ui()
        self.try_load_default_model()

    def set_window_icon(self):
        if os.path.exists(FAVICON_PATH):
            try:
                self.setWindowIcon(QIcon(FAVICON_PATH))
            except Exception as e:
                print(f"设置图标失败: {e}")
        else:
            print(f"警告：未找到图标文件 {FAVICON_PATH}")

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        load_action = file_menu.addAction("加载模型...")
        load_action.triggered.connect(self.load_model_dialog)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # 图片检测标签页
        self.img_tab = QWidget()
        tabs.addTab(self.img_tab, "图片检测")
        self.setup_image_tab()

        # 视频检测标签页
        self.video_tab = QWidget()
        tabs.addTab(self.video_tab, "视频检测")
        self.setup_video_tab()

        self.status_label = QLabel("模型未加载")
        self.statusBar().addWidget(self.status_label)

    def setup_image_tab(self):
        layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.img_path_edit = QLineEdit()
        self.img_path_edit.setPlaceholderText("请选择图片文件...")
        self.img_browse_btn = QPushButton("浏览...")
        self.img_browse_btn.clicked.connect(self.browse_image)
        file_layout.addWidget(self.img_path_edit)
        file_layout.addWidget(self.img_browse_btn)
        layout.addLayout(file_layout)

        self.img_preview_label = QLabel()
        self.img_preview_label.setAlignment(Qt.AlignCenter)
        self.img_preview_label.setMinimumHeight(200)
        self.img_preview_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.img_preview_label)

        self.img_detect_btn = QPushButton("检测")
        self.img_detect_btn.clicked.connect(self.detect_image)
        layout.addWidget(self.img_detect_btn)

        self.img_result_text = QTextEdit()
        self.img_result_text.setReadOnly(True)
        self.img_result_text.setMaximumHeight(100)
        layout.addWidget(self.img_result_text)

        self.img_tab.setLayout(layout)

    def setup_video_tab(self):
        layout = QVBoxLayout()

        # 文件选择
        file_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("请选择视频文件...")
        self.video_browse_btn = QPushButton("浏览...")
        self.video_browse_btn.clicked.connect(self.browse_video)
        file_layout.addWidget(self.video_path_edit)
        file_layout.addWidget(self.video_browse_btn)
        layout.addLayout(file_layout)

        # 视频预览
        self.video_preview_label = QLabel()
        self.video_preview_label.setAlignment(Qt.AlignCenter)
        self.video_preview_label.setMinimumHeight(150)
        self.video_preview_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.video_preview_label)

        # 抽帧设置区域
        group_box = QGroupBox("抽帧设置")
        group_layout = QVBoxLayout()

        # 智能抽帧 + 手动输入框
        smart_layout = QHBoxLayout()
        self.smart_checkbox = QCheckBox("智能抽帧")
        self.smart_checkbox.setChecked(True)
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(1, 100)
        self.spin_frames.setValue(10)
        self.spin_frames.setEnabled(False)  # 初始智能模式，禁用
        smart_layout.addWidget(self.smart_checkbox)
        smart_layout.addWidget(self.spin_frames)
        group_layout.addLayout(smart_layout)

        # 抽取全部帧复选框
        self.full_checkbox = QCheckBox("抽取全部帧")
        self.full_checkbox.setChecked(False)
        group_layout.addWidget(self.full_checkbox)

        # 预计帧数提示
        self.expected_label = QLabel("")
        group_layout.addWidget(self.expected_label)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        # 信号连接
        self.smart_checkbox.stateChanged.connect(self.on_smart_toggled)
        self.full_checkbox.stateChanged.connect(self.on_full_toggled)
        self.video_path_edit.textChanged.connect(self.update_expected_frames)
        self.spin_frames.valueChanged.connect(self.update_expected_frames)

        # 检测按钮和进度条
        btn_progress_layout = QHBoxLayout()
        self.video_detect_btn = QPushButton("检测视频")
        self.video_detect_btn.clicked.connect(self.detect_video)
        btn_progress_layout.addWidget(self.video_detect_btn)
        self.video_progress = QProgressBar()
        self.video_progress.setVisible(False)
        btn_progress_layout.addWidget(self.video_progress)
        layout.addLayout(btn_progress_layout)

        # 结果表格
        self.video_table = QTableWidget()
        self.video_table.setColumnCount(3)
        self.video_table.setHorizontalHeaderLabels(["帧序号", "结果", "得分"])
        self.video_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.video_table)

        # 统计信息
        self.video_stats_label = QLabel("等待检测...")
        layout.addWidget(self.video_stats_label)

        self.video_tab.setLayout(layout)

    def on_smart_toggled(self):
        if self.smart_checkbox.isChecked():
            self.spin_frames.setEnabled(False)
        else:
            self.spin_frames.setEnabled(not self.full_checkbox.isChecked())
        self.update_expected_frames()

    def on_full_toggled(self):
        if self.full_checkbox.isChecked():
            self.smart_checkbox.setEnabled(False)
            self.spin_frames.setEnabled(False)
        else:
            self.smart_checkbox.setEnabled(True)
            # 智能抽帧状态下，手动输入框禁用；否则启用
            self.spin_frames.setEnabled(not self.smart_checkbox.isChecked())
        self.update_expected_frames()

    def update_expected_frames(self):
        video_path = self.video_path_edit.text().strip()
        if not video_path or not os.path.exists(video_path):
            self.expected_label.setText("")
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            self.expected_label.setText("无法获取视频信息")
            return

        if self.full_checkbox.isChecked():
            self.expected_label.setText(f"将抽取全部 {total_frames} 帧")
        elif self.smart_checkbox.isChecked():
            # 智能抽帧策略：若总帧数≤30则全抽，否则抽30帧
            num = max(1, min(30, total_frames))
            self.expected_label.setText(f"将智能抽取 {num} 帧")
        else:
            manual_num = self.spin_frames.value()
            if manual_num > total_frames:
                self.expected_label.setText(
                    f"注意：抽帧数({manual_num})大于总帧数({total_frames})，将抽取全部 {total_frames} 帧")
            else:
                self.expected_label.setText(f"将手动抽取 {manual_num} 帧")

    def try_load_default_model(self):
        if os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(DEFAULT_SCALER_PATH) and os.path.exists(
                DEFAULT_THRESHOLD_PATH):
            try:
                with open(DEFAULT_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(DEFAULT_SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(DEFAULT_THRESHOLD_PATH, 'r') as f:
                    self.threshold = float(f.read().strip())
                self.status_label.setText(f"模型已加载，阈值: {self.threshold:.6f}")
                QMessageBox.information(self, "成功", f"默认模型加载成功！阈值 = {self.threshold:.6f}")
            except Exception as e:
                self.status_label.setText("默认模型加载失败")
                QMessageBox.warning(self, "警告", f"默认模型加载失败: {str(e)}")
        else:
            self.status_label.setText("未找到默认模型，请手动加载")

    def load_model_dialog(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pkl)")
        if not model_file:
            return
        base_dir = os.path.dirname(model_file)
        base_name = os.path.basename(model_file).replace(".pkl", "")

        scaler_file = os.path.join(base_dir, f"{base_name}_scaler.pkl")
        threshold_file = os.path.join(base_dir, f"{base_name}_threshold.txt")

        if not os.path.exists(scaler_file):
            scaler_file, _ = QFileDialog.getOpenFileName(self, "选择标准化器文件", base_dir, "标准化器 (*.pkl)")
            if not scaler_file:
                return
        if not os.path.exists(threshold_file):
            threshold_file, _ = QFileDialog.getOpenFileName(self, "选择阈值文件", base_dir, "阈值文件 (*.txt)")
            if not threshold_file:
                return

        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(threshold_file, 'r') as f:
                self.threshold = float(f.read().strip())
            self.status_label.setText(f"模型已加载，阈值: {self.threshold:.6f}")
            QMessageBox.information(self, "成功", f"模型加载成功！阈值 = {self.threshold:.6f}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.img_path_edit.setText(path)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(self.img_preview_label.width(), 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.img_preview_label.setPixmap(scaled)
            else:
                self.img_preview_label.clear()

    def detect_image(self):
        if not self.check_model_loaded():
            return
        img_path = self.img_path_edit.text().strip()
        if not img_path or not os.path.exists(img_path):
            QMessageBox.warning(self, "警告", "请先选择有效的图片文件")
            return

        self.img_detect_btn.setEnabled(False)
        self.img_result_text.clear()

        self.img_thread = ImageDetectionThread(img_path, self.model, self.scaler, self.threshold)
        self.img_thread.finished.connect(self.on_image_detected)
        self.img_thread.start()

    def on_image_detected(self, result):
        pred, score, msg = result
        if pred is None:
            self.img_result_text.append(f"检测失败: {msg}")
        else:
            self.img_result_text.append(f"检测结果: {pred}")
            self.img_result_text.append(f"异常得分: {score:.6f} (阈值: {self.threshold:.6f})")
        self.img_detect_btn.setEnabled(True)

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path_edit.setText(path)
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img)
                scaled = pixmap.scaled(self.video_preview_label.width(), 150, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)
                self.video_preview_label.setPixmap(scaled)
            else:
                self.video_preview_label.clear()

    def detect_video(self):
        if not self.check_model_loaded():
            return
        video_path = self.video_path_edit.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "警告", "请先选择有效的视频文件")
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            QMessageBox.warning(self, "警告", "无法获取视频帧数")
            return

        if self.full_checkbox.isChecked():
            # 抽取全部帧
            num_frames = total_frames
            indices = list(range(total_frames))
        elif self.smart_checkbox.isChecked():
            # 智能抽帧
            num_frames = max(1, min(30, total_frames))
            indices = sorted(random.sample(range(total_frames), num_frames))
        else:
            # 手动抽帧
            num_frames = self.spin_frames.value()
            if num_frames > total_frames:
                num_frames = total_frames
                QMessageBox.information(self, "提示", f"抽帧数超过视频总帧数，将抽取全部 {total_frames} 帧")
            indices = sorted(random.sample(range(total_frames), num_frames))

        # 清空表格和统计
        self.video_table.setRowCount(0)
        self.video_stats_label.setText("检测中...")
        self.video_progress.setVisible(True)
        self.video_progress.setMaximum(num_frames)
        self.video_progress.setValue(0)
        self.video_detect_btn.setEnabled(False)

        self.video_thread = VideoDetectionThread(video_path, indices, self.model, self.scaler, self.threshold)
        self.video_thread.progress.connect(self.video_progress.setValue)
        self.video_thread.frame_result.connect(self.add_video_table_row)
        self.video_thread.finished.connect(self.on_video_detected)
        self.video_thread.start()

    def add_video_table_row(self, idx, pred, score):
        row = self.video_table.rowCount()
        self.video_table.insertRow(row)
        self.video_table.setItem(row, 0, QTableWidgetItem(str(idx)))
        self.video_table.setItem(row, 1, QTableWidgetItem(pred))
        self.video_table.setItem(row, 2, QTableWidgetItem(f"{score:.6f}" if score != 0 else "-"))

    def on_video_detected(self, results, stats):
        self.video_progress.setVisible(False)
        self.video_detect_btn.setEnabled(True)
        total = len(results)

        # 计算视频整体结论（多数投票法）
        if stats['real'] + stats['fake'] == 0:
            video_verdict = "无有效人脸帧，无法判断"
        else:
            video_verdict = "伪造" if stats['fake'] > stats['real'] else "真实"
        stats_text = (f"总检测帧: {total}  "
                      f"真实: {stats['real']}  "
                      f"伪造: {stats['fake']}  "
                      f"无人脸: {stats['no_face']}\n"
                      f"视频整体结论: {video_verdict}")
        self.video_stats_label.setText(stats_text)

    def check_model_loaded(self):
        if self.model is None or self.scaler is None or self.threshold is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return False
        return True

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 也可以给应用程序全局设置图标（可选）
    if os.path.exists(FAVICON_PATH):
        app.setWindowIcon(QIcon(FAVICON_PATH))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
