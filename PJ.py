import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import copy
import time
import cv2
import mediapipe as mp

# ⚙️ Config
train_root = '/home/yunddu/Semicon_Project/npy_data/train'
val_root = '/home/yunddu/Semicon_Project/npy_data/val'
video_folder = '/home/yunddu/Semicon_Project/Dataset/train/daily'
output_image_path = 'sample_visualization.png'

batch_size = 16
num_epochs = 50
learning_rate = 0.001
early_stopping_patience = 10

label_mapping = {'daily': 0, 'falldown': 1}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ✅ Dataset
class SkeletonDataset(Dataset):
    def __init__(self, root_dir, label_mapping, sequence_length=30, transform=None):
        self.samples = []
        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name.strip())
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith('.npy'):
                        self.samples.append((os.path.join(label_path, file), label_mapping[label_name.strip()]))
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        sequence = np.load(file_path)
        if self.transform:
            sequence = self.transform(sequence)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02): self.mean = mean; self.std = std
    def __call__(self, sequence): return sequence + np.random.normal(self.mean, self.std, sequence.shape)

# ✅ Model
class LSTMActionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# ✅ DataLoader
train_loader = DataLoader(SkeletonDataset(train_root, label_mapping, transform=AddGaussianNoise()),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SkeletonDataset(val_root, label_mapping),
                        batch_size=batch_size, shuffle=False)

# ✅ Training Setup
model = LSTMActionClassifier(input_dim=66, hidden_dim=128, num_layers=2, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = np.inf
early_stopping_counter = 0
start_time = time.time()

# ✅ Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            val_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        print(f"✅ Saving best model (Val Loss improved: {best_val_loss:.4f} -> {val_loss:.4f})")
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        print(f"EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")
        if early_stopping_counter >= early_stopping_patience:
            print("⛔ EarlyStopping triggered!")
            break

training_time = time.time() - start_time
torch.save(best_model_wts, 'best_model.pth')
print("✅ Best model saved as best_model.pth")

# ✅ Evaluation & Visualization
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
all_labels, all_preds, all_probs = [], [], []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        probs = softmax(outputs)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping.keys())
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Validation)')
plt.show()

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
val_accuracy = 100 * np.mean(np.array(all_labels) == np.array(all_preds))

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Accuracy')
plt.text(0.6, 0.25, f'Val Acc: {val_accuracy:.2f}%\nTime: {training_time:.1f}s',
         bbox=dict(facecolor='white', alpha=0.7))
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ✅ 예시 영상 시각화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

try:
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    sample_video = os.path.join(video_folder, video_files[0])
    cap = cv2.VideoCapture(sample_video)
    ret, frame = cap.read()
    cap.release()
    if ret:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(output_image_path, frame)
        print(f"[📷] 시각화 결과 저장됨: {output_image_path}")
except Exception as e:
    print(f"[⚠️] 시각화 실패: {e}")

# ✅ .npy 데이터 예시 출력
try:
    npy_sample = os.path.join(train_root, 'daily', sorted(os.listdir(os.path.join(train_root, 'daily')))[0])
    data = np.load(npy_sample)
    print(f"[📐] .npy shape: {data.shape}")
    print(f"[🔍] 첫 프레임 keypoint 값:\n{data[0]}")
except Exception as e:
    print(f"[⚠️] .npy 로드 실패: {e}")
