import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------------------------------
# 1. LOAD BERT EMBEDDINGS + CSV
# ------------------------------------------------------
X_text = torch.load("bertweet_embeddings.pt").numpy()   # shape: (N, 768)
df = pd.read_csv("cleaned_tweets.csv")

y = df["virality"].astype(int).values

print("Text embedding shape:", X_text.shape)
print("y shape:", y.shape)

# ------------------------------------------------------
# 2. SELECT METADATA FEATURES
# ------------------------------------------------------
metadata_cols = [
    "followers", "favs", "retweets", "quotes",
    "negative", "positive", "sentimentPosNeg"
]

metadata = df[metadata_cols].copy()

# convert to numeric
metadata = metadata.apply(pd.to_numeric, errors="coerce").fillna(0)

# scale metadata
scaler = StandardScaler()
metadata_scaled = scaler.fit_transform(metadata.values)

print("Metadata shape:", metadata_scaled.shape)

# ------------------------------------------------------
# 3. CONCATENATE TEXT EMBEDDINGS + METADATA
# ------------------------------------------------------
X_full = np.concatenate([X_text, metadata_scaled], axis=1)

print("Final combined feature shape:", X_full.shape)
# should be (N, 768 + num_metadata)

# ------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ------------------------------------------------------
# 5. TORCH TENSORS
# ------------------------------------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ------------------------------------------------------
# 6. CLASS WEIGHTS (IMBALANCE HANDLING)
# ------------------------------------------------------
class_counts = np.bincount(y_train)
total = len(y_train)
weights = torch.tensor(total / (3 * class_counts), dtype=torch.float32)

print("Class weights:", weights)

# ------------------------------------------------------
# 7. DEFINE MODEL (LOGISTIC REGRESSION)
# ------------------------------------------------------
input_dim = X_train.shape[1]   # 768 + metadata count

model = nn.Linear(input_dim, 3)

loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------
# 8. TRAINING LOOP WITH TIME TRACKING
# ------------------------------------------------------
epochs = 8
epoch_times = []

print("\nStarting training...\n")
start_total = time.time()

for epoch in range(epochs):
    start_epoch = time.time()

    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = loss_fn(logits, y_train_t)
    loss.backward()
    optimizer.step()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    avg_so_far = np.mean(epoch_times)
    epochs_left = epochs - (epoch + 1)
    eta = avg_so_far * epochs_left

    print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f} — "
          f"Epoch time: {epoch_time:.2f} sec — ETA: {eta/60:.2f} min")

total_time = time.time() - start_total
print(f"\nTotal training time: {total_time/60:.2f} minutes")

# ------------------------------------------------------
# 9. EVALUATION
# ------------------------------------------------------
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)
    accuracy = (preds == y_test_t).float().mean().item()

print("\nTest accuracy:", accuracy)

preds_np = preds.numpy()
y_test_np = y_test_t.numpy()

print("\nClassification Report (Macro & Per-Class):")
print(classification_report(y_test_np, preds_np, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_np, preds_np))

# ------------------------------------------------------
# 10. DUMMY RANDOM BASELINE
# ------------------------------------------------------
np.random.seed(0)
random_preds = np.random.choice([0,1,2], size=len(y_test))
print("\nRandom baseline accuracy:", accuracy_score(y_test, random_preds))

