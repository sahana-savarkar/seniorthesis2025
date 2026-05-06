import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

df = pd.read_csv("cleaned_tweets.csv")
tweets = df["text"].tolist()

# load BERTweet
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base").to(DEVICE)
model.eval()

all_embeddings = []

for tweet in tqdm(tweets):
    inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        mean_embedding = token_embeddings.mean(dim=1)

    all_embeddings.append(mean_embedding.cpu())

# stacking embeddings into one tensor
all_embeddings = torch.vstack(all_embeddings)
torch.save(all_embeddings, "bertweet_embeddings.pt")

print("Saved embeddings to bertweet_embeddings.pt")
