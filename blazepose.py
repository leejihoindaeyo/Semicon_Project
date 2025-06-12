import cv2
import mediapipe as mp
import os
import glob

# 🔧 입력 및 출력 디렉토리 설정
video_base_dir = '/Users/ijiho/Documents/person_detect/DataSet/train'
output_root = '/Users/ijiho/Documents/person_detect/output_frames'
os.makedirs(output_root, exist_ok=True)

# 🔧 MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# 🔁 모든 클래스(daily, falldown) 폴더 반복
for label_name in os.listdir(video_base_dir):
    label_path = os.path.join(video_base_dir, label_name)
    if not os.path.isdir(label_path):
        continue

    video_files = glob.glob(os.path.join(label_path, '*.mp4'))

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ 동영상 열기 실패:", video_path)
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_root, label_name, video_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"📹 처리 중: {label_name}/{video_name}")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            save_path = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(save_path, frame)
            frame_count += 1

        cap.release()
        print(f"✅ 완료: {label_name}/{video_name} → {frame_count}프레임 저장됨")

pose.close()

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from collections import Counter

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# 최대 프레임 수 설정 (예: 10초 @30fps)
MAX_FRAMES = 300
KEYPOINT_DIM = 132  # 33개 관절 × (x, y, z, visibility)

# 경로 설정
base_dir = '/Users/ijiho/Documents/person_detect/DataSet/train'
output_dir = '/Users/ijiho/Documents/person_detect/npy_data'
os.makedirs(output_dir, exist_ok=True)

X_data = []
y_data = []

label_dict = {'daily': 0, 'falldown': 1}

for label_name, label_num in label_dict.items():
    class_dir = os.path.join(base_dir, label_name)
    video_files = glob.glob(os.path.join(class_dir, '*.mp4'))

    print(f"\n📂 처리 중 클래스: {label_name} ({label_num}) - 총 {len(video_files)}개 파일")

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        keypoints_seq = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                frame_keypoints = []
                for lm in results.pose_landmarks.landmark:
                    frame_keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                keypoints_seq.append(frame_keypoints)

        cap.release()

        if len(keypoints_seq) == 0:
            print(f"⚠️ 관절 감지 실패: {os.path.basename(video_path)}")
            continue

        keypoints_seq = np.array(keypoints_seq)

        if keypoints_seq.shape[0] >= MAX_FRAMES:
            keypoints_seq = keypoints_seq[:MAX_FRAMES]
        else:
            padding = np.zeros((MAX_FRAMES - keypoints_seq.shape[0], KEYPOINT_DIM))
            keypoints_seq = np.vstack((keypoints_seq, padding))

        X_data.append(keypoints_seq)
        y_data.append(label_num)

        print(f"✅ 저장됨: {os.path.basename(video_path)} → shape {keypoints_seq.shape}")

pose.close()

# numpy array로 변환
X_data = np.array(X_data)  # shape: (N, 300, 132)
y_data = np.array(y_data)  # shape: (N,)

# ✅ 클래스별 저장 개수 출력
label_counts = Counter(y_data)
print("\n📊 클래스별 저장된 샘플 수:")
for label, count in label_counts.items():
    label_name = [k for k, v in label_dict.items() if v == label][0]
    print(f"  • {label_name} ({label}): {count}개")

print(f"\n📁 총 저장된 샘플 수: {len(X_data)}개")
print(f"📐 배열 크기: {X_data.shape}")

# ✅ 최종 저장
np.save(os.path.join(output_dir, 'X.npy'), X_data)
np.save(os.path.join(output_dir, 'y.npy'), y_data)
print("\n💾 X.npy와 y.npy 저장 완료")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import Counter

# ✅ LSTM 모델 정의
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_layers=1, num_classes=2):
        super(FallDetectionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# ✅ Confusion Matrix 함수
def plot_confusion_matrix(y_true, y_pred, class_names=["daily", "falldown"]):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

# ✅ 평가 함수
def evaluate_model(model, data_loader, device, set_name="Test"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    print(f"\n📄 {set_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["daily", "falldown"]))

    plot_confusion_matrix(all_labels, all_preds, class_names=["daily", "falldown"])

# ✅ 데이터 불러오기
X = np.load('/Users/ijiho/Documents/person_detect/npy_data/X.npy')  # (N, 300, 132)
y = np.load('/Users/ijiho/Documents/person_detect/npy_data/Y.npy')  # (N,)

# 텐서 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ✅ 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallDetectionLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ✅ 학습 루프
num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"📘 Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

# ✅ 학습 손실 시각화
plt.figure(figsize=(8, 4))
plt.plot(train_losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Train/ Test 평가 및 시각화
evaluate_model(model, train_loader, device, set_name="Train")
evaluate_model(model, test_loader, device, set_name="Test")
