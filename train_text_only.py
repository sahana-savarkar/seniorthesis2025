import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# load data
print("Loading cleaned dataset...")
df = pd.read_csv("cleaned_tweets.csv")

texts = df["clean_text"].astype(str).tolist()
labels = df["virality"].astype(int).tolist()

# load tokenizer
print("Loading BERTweet tokenizer + model...")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
bert = AutoModel.from_pretrained("vinai/bertweet-base")


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=64,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }


dataset = TweetDataset(texts, labels, tokenizer)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("DataLoader ready.")

# MODEL
class BERTweetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert

        # freezing BERT parameters to speed up training
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # zero out the gradients; speed things up 
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_emb = output.last_hidden_state[:, 0, :]
        x = self.dropout(cls_emb)
        return self.classifier(x)

model = BERTweetClassifier().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# TRAIN
EPOCHS = 3
print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} complete — Avg Loss: {total_loss / len(train_loader):.4f}")


# SAVE
torch.save(model.state_dict(), "models/bertweet_text_only.pt")
print("\nSaved model → models/bertweet_text_only.pt")
