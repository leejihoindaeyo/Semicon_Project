import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import copy

# ⚙️ Config
train_root = '/home/yunddu/Semicon_Project/npy_data/train'  # 수정
val_root = '/home/yunddu/Semicon_Project/npy_data/val'      # 수정

batch_size = 16
num_epochs = 30
learning_rate = 0.001
early_stopping_patience = 5

# ✅ Dataset + DataLoader + Transforms
class SkeletonDataset(Dataset):
    def __init__(self, root_dir, label_mapping, sequence_length=30, transform=None):
        self.root_dir = root_dir
        self.label_mapping = label_mapping
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []

        for label_name in os.listdir(root_dir):
            label_name = label_name.strip()  # 공백 제거 (안정성 ↑)
            label_path = os.path.join(root_dir, label_name)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith('.npy'):
                        full_path = os.path.join(label_path, file)
                        self.samples.append((full_path, label_mapping[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        sequence = np.load(file_path)

        if self.transform:
            sequence = self.transform(sequence)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, sequence):
        noise = np.random.normal(self.mean, self.std, sequence.shape)
        return sequence + noise

# ✅ LSTM 모델 정의
class LSTMActionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMActionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ✅ Setup
label_mapping = {
    'daily': 0,
    'falldown': 1
}

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DataLoaders
train_dataset = SkeletonDataset(train_root, label_mapping, sequence_length=30, transform=AddGaussianNoise())
val_dataset = SkeletonDataset(val_root, label_mapping, sequence_length=30, transform=None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model, Loss, Optimizer, Scheduler
input_dim = 66
hidden_dim = 128
num_layers = 2
num_classes = 2

model = LSTMActionClassifier(input_dim, hidden_dim, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ✅ Training loop + EarlyStopping + Checkpoint
best_val_loss = np.inf
early_stopping_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

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

# ✅ Save best model
torch.save(best_model_wts, 'best_model.pth')
print("✅ Best model saved as best_model.pth")

# ✅ Confusion Matrix
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Validation)')
plt.show()

# ✅ ROC Curve
softmax = nn.Softmax(dim=1)

all_labels_roc = []
all_probs = []
all_preds_roc = []

with torch.no_grad():
    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        probs = softmax(outputs)[:, 1]  # class 1 (falldown)의 확률만 가져옴

        all_labels_roc.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds_roc.extend(torch.argmax(outputs, 1).cpu().numpy())

# Compute ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels_roc, all_probs)
roc_auc = auc(fpr, tpr)

# ✅ Plot ROC + Accuracy 표시
fig, ax = plt.subplots(figsize=(8, 6))

# ROC Curve
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve + Validation Accuracy')

# ✅ Accuracy 계산
val_accuracy_final = 100 * np.mean(np.array(all_labels_roc) == np.array(all_preds_roc))

# Accuracy 텍스트 추가
ax.text(0.6, 0.2, f'Val Accuracy = {val_accuracy_final:.2f}%', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7))

ax.legend(loc='lower right')
plt.grid()
plt.show()
