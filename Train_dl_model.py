import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

from model_def import OutcomeClassifier

# -----------------------------
# CONFIG — change path to match your project
# -----------------------------
DATA_PATH = "Court_dataset.csv"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good enough for a project
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
df = df.fillna("")
df = df[df["outcome"].notna() & (df["outcome"] != "") & (df["outcome"] != "nan")]

df["combined_text"] = (
    df["name"].astype(str) + " " +
    df["case_type"].astype(str) + " " +
    df["case_category"].astype(str) + " " +
    df.get("judge", "").astype(str) + " " +
    df.get("status", "").astype(str)
)

# -----------------------------
# SENTENCE-BERT EMBEDDINGS (this replaces TF-IDF as input)
# -----------------------------
print("[INFO] Loading Sentence-BERT model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("[INFO] Generating embeddings for all cases...")
X = embed_model.encode(df["combined_text"].tolist(), show_progress_bar=True)

# -----------------------------
# LABEL ENCODING
# -----------------------------
le_out = LabelEncoder()
y = le_out.fit_transform(df["outcome"])
num_classes = len(le_out.classes_)

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class CaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(CaseDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# MODEL / LOSS / OPTIMIZER
# -----------------------------
model = OutcomeClassifier(input_dim=X_train.shape[1], hidden_dim=128, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAIN LOOP
# -----------------------------
train_losses, val_accuracies = [], []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        preds = torch.argmax(test_outputs, dim=1)
        acc = accuracy_score(y_test, preds)

    train_losses.append(epoch_loss / len(train_loader))
    val_accuracies.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | Val Accuracy: {acc:.4f}")

# -----------------------------
# FINAL EVALUATION (for your project report — compare vs RandomForest)
# -----------------------------
model.eval()
with torch.no_grad():
    final_preds = torch.argmax(model(X_test), dim=1)

print("\n[FINAL CLASSIFICATION REPORT]")
print(classification_report(y_test, final_preds, target_names=le_out.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))

# -----------------------------
# SAVE MODEL + ARTIFACTS (app.py loads these)
# -----------------------------
torch.save(model.state_dict(), "dl_model.pt")
pickle.dump(le_out, open("dl_label_encoder.pkl", "wb"))
pickle.dump(
    {"input_dim": X_train.shape[1], "num_classes": num_classes, "embed_model_name": EMBED_MODEL_NAME},
    open("dl_model_config.pkl", "wb")
)

print("\n✅ Deep learning model trained and saved: dl_model.pt, dl_label_encoder.pkl, dl_model_config.pkl")