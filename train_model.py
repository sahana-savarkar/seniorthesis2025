# train_model.py (very short version)

import torch # core PyTorch library for tensors and computation.
import torch.nn as nn # neural network module (layers, loss functions, etc.).
import pandas as pd # loads CSV files (your dataset).
from sklearn.model_selection import train_test_split # splits data into training and testing sets.

# Load data
embeddings = torch.load("data/embeddings.pt")          # (N, 768)
df = pd.read_csv("data/data_with_labels.csv")          # contains virality + metadata
y = torch.tensor(df["virality"].values).long()         # (N,)

# Optionally add numeric metadata (comment out if not needed)
numeric = torch.tensor(df[["likes", "retweets", "hashtags"]].values).float()
X = torch.cat((embeddings.float(), numeric), dim=1)    # concatenate our 768-dimensional embedding w/ 3 numeric features

# Train/test split from sklearn 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Simple neural network
model = nn.Sequential(
    nn.Linear(X.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 3)      # 3 virality classes: 0/1/2
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(15):
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/15 — Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    accuracy = (preds == y_test).float().mean().item()

print(f"\n Final accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "models/virality_classifier.pt")
print("Saved → models/virality_classifier.pt")