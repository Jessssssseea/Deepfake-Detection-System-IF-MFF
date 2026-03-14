import os
import cv2
import dlib
import numpy as np
import shutil
from imutils import face_utils
import argparse
import time
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ====================== 配置参数 ======================
DEFAULT_INPUT_DIR = r"D:\AI face\data_min\fake"
DEFAULT_OUTPUT_DIR = r"D:\AI face\data_min\dedup_fake"
DEFAULT_ERROR_DIR = r"D:\AI face\data_min\error_images_fake"  # 出错+无人脸都放这里
DEFAULT_NUM_KEEP = 240000
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SIMILARITY_THRESHOLD = 0.8  # 参数保留但不再使用
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
# ======================================================

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

error_image_paths = []  # 所有失败图片：读失败、无人脸、报错

def compute_quality_and_embedding(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            error_image_paths.append(image_path)
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        if len(faces) == 0:
            # 无人脸 → 也算出错
            error_image_paths.append(image_path)
            return None

        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # 质量评分
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0

        left_eye = np.mean(shape[36:42], axis=0)
        right_eye = np.mean(shape[42:48], axis=0)
        nose_tip = shape[30]
        eye_center = (left_eye + right_eye) / 2
        offset_x = nose_tip[0] - eye_center[0]
        face_width = face.right() - face.left()
        pose_score = 1.0 - abs(offset_x) / (face_width / 2) if face_width > 0 else 0

        sym_pairs = [(36,45), (39,42), (48,54), (31,35)]
        sym_diff_sum = 0
        for i, j in sym_pairs:
            dist_left = np.linalg.norm(shape[i] - shape[27])
            dist_right = np.linalg.norm(shape[j] - shape[27])
            sym_diff_sum += abs(dist_left - dist_right)
        sym_score = 1.0 / (1.0 + sym_diff_sum)

        img_area = img.shape[0] * img.shape[1]
        face_area = face.width() * face.height()
        area_ratio = face_area / img_area

        def eye_aspect_ratio(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(shape[36:42])
        right_ear = eye_aspect_ratio(shape[42:48])
        ear = (left_ear + right_ear) / 2.0
        ear_score = min(ear / 0.3, 1.0) if ear < 0.3 else 1.0

        score = (0.3 * (laplacian_var / 1000) +
                 0.2 * brightness_score +
                 0.2 * pose_score +
                 0.1 * sym_score +
                 0.1 * area_ratio +
                 0.1 * ear_score)

        try:
            face_chip = dlib.get_face_chip(img, shape, size=160)
        except:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_crop = img[y1:y2, x1:x2]
            face_chip = cv2.resize(face_crop, (160, 160))

        face_chip = face_chip.astype(np.float32) / 255.0
        face_chip = (face_chip - 0.5) / 0.5
        face_tensor = torch.tensor(face_chip).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy().flatten()

        return (score, embedding, image_path)

    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        error_image_paths.append(image_path)
        return None

def main():
    parser = argparse.ArgumentParser(description='筛选高质量的人脸图片（已移除重复过滤）')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--error', type=str, default=DEFAULT_ERROR_DIR)
    parser.add_argument('--num', type=int, default=DEFAULT_NUM_KEEP)
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD)  # 保留但不使用
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    error_dir = args.error
    num_keep = args.num
    # sim_thresh = args.threshold   # 不再使用

    if not os.path.exists(input_dir):
        print(f"输入文件夹不存在: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(image_exts):
                image_paths.append(os.path.join(root, f))

    print(f"共找到 {len(image_paths)} 张图片")

    if len(image_paths) == 0:
        print("没有图片可处理")
        return

    results = []
    start_time = time.time()
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"已处理 {i}/{len(image_paths)} 张...")
        res = compute_quality_and_embedding(path)
        if res is not None:
            results.append(res)

    print(f"处理完成，有效图片数：{len(results)}，耗时：{time.time()-start_time:.1f}s")

    # 修改部分：直接按质量分数排序并取前 num_keep 张，不再去重
    kept_paths = []
    if len(results) > 0:
        results.sort(key=lambda x: x[0], reverse=True)  # 按分数降序
        # 取前 num_keep 张图片的路径
        kept_paths = [path for _, _, path in results[:num_keep]]
        print(f"按质量排序后共保留 {len(kept_paths)} 张图片")

        for i, path in enumerate(kept_paths):
            filename = os.path.basename(path)
            new_filename = f"{i+1:06d}_{filename}"
            dst = os.path.join(output_dir, new_filename)
            shutil.move(path, dst)
            if (i+1) % 500 == 0:
                print(f"已移动 {i+1}/{len(kept_paths)}")

    # 移动所有出错图片：读失败 + 无人脸 + 程序报错
    if error_image_paths:
        print(f"\n开始移动 {len(error_image_paths)} 张无效图片到错误文件夹...")
        for i, path in enumerate(error_image_paths):
            try:
                if os.path.exists(path):
                    fname = os.path.basename(path)
                    dst = os.path.join(error_dir, fname)
                    shutil.move(path, dst)
            except Exception as e:
                print(f"移动失败 {path}: {e}")
        print(f"无效图片已全部移动至：{error_dir}")

    print("\n全部完成！")

if __name__ == '__main__':
    main()