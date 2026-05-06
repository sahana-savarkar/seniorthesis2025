import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Load embeddings + labels
X = torch.load("bertweet_embeddings.pt").numpy()
df = pd.read_csv("cleaned_tweets.csv")
y = df["virality"].astype(int).values

print("X shape:", X.shape)
print("y shape:", y.shape)

# Split data — STRATIFIED because classes are imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

import torch.nn as nn
import torch.optim as optim
import numpy as np

# Convert numpy → torch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Compute class weights
class_counts = np.bincount(y_train)
total = len(y_train)

weights = total / (3 * class_counts)
weights = torch.tensor(weights, dtype=torch.float32)

print("Class weights:", weights)

# Define logistic regression model
model = nn.Linear(768, 3)   # W: (3,768), b: (3,)
# model = nn.Sequential(
#     nn.Linear(768, 256),
#     nn.ReLU(),
#     nn.Dropout(0.2),

#     nn.Linear(256, 128),
#     nn.ReLU(),
#     nn.Dropout(0.1),

#     nn.Linear(128, 3)
# )


loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

import time
epochs = 13
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

    # compute ETA
    avg_so_far = np.mean(epoch_times)
    epochs_left = epochs - (epoch + 1)
    eta = avg_so_far * epochs_left

    print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f} — "
          f"Epoch time: {epoch_time:.2f} sec — "
          f"ETA: {eta/60:.2f} min")

total_time = time.time() - start_total
print(f"\nTotal training time: {total_time/60:.2f} minutes")


# Training loop
# epochs = 100
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     logits = model(X_train_t)
#     loss = loss_fn(logits, y_train_t)
#     loss.backward()
#     optimizer.step()
    
#     print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)
    accuracy = (preds == y_test_t).float().mean().item()

print("\nTest accuracy:", accuracy)

from sklearn.metrics import classification_report, confusion_matrix

preds_np = preds.numpy()
y_test_np = y_test_t.numpy()

print("\nClassification Report (Macro & Per-Class):")
print(classification_report(y_test_np, preds_np, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_np, preds_np))

##### DUMMY CASE #####

import numpy as np
from sklearn.metrics import accuracy_score
np.random.seed(0)
random_preds = np.random.choice([0,1,2], size=len(y_test))
print("Random baseline accuracy:", accuracy_score(y_test, random_preds))


