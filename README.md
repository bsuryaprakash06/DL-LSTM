# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

Problem Statement:

The objective is to develop a Deep Learning model that can automatically identify and classify Named Entities (such as names of people, locations, organizations, etc.) within a given sentence. By using a Bidirectional LSTM (BiLSTM), the model learns to understand the context of a word based on both the words that come before it and the words that come after it.

<img width="336" height="381" alt="image" src="https://github.com/user-attachments/assets/41d4dbdd-c2a3-4929-85b9-d33cd0572a2c" />

## Theory
1. Named Entity Recognition (NER)
NER is a Natural Language Processing (NLP) task. It involves processing a sentence and labeling each word with a specific category (e.g., B-per for a person's name, B-geo for a location). If a word doesn't belong to a category, it is usually labeled as 'O' (Other).

2. Bidirectional LSTM (BiLSTM)
A standard LSTM processes text from left to right. A Bidirectional LSTM processes the sentence in two directions:

    Forward: To understand the context from the start of the sentence.
    
    Backward: To understand the context from the end of the sentence.
    This allows the model to have a "full view" of the sentence, which is essential for correctly identifying entities.

3. Embedding Layer
Since computers cannot understand text directly, we use an Embedding Layer. This converts each unique word into a dense vector (a list of numbers) that represents its meaning. Words with similar meanings or usages will have similar vectors.

4. Padding
Sentences in a dataset have different lengths. However, neural networks require inputs to be the same size. Padding adds special "empty" tokens (like zeros) to shorter sentences so that every input in a batch has a uniform length.

5. Cross-Entropy Loss
In NER, we are trying to classify each word into one of many tag categories. We use Cross-Entropy Loss to measure how well the predicted tags match the actual tags. The training process adjusts the model to minimize this loss.

## DESIGN STEPS

### Step 1: Load Dataset

* Import required libraries.
* Load the NER dataset from CSV file.
* Handle missing values and group words into sentences.

### Step 2: Convert Words and Tags to Numbers

* Create word-to-index and tag-to-index dictionaries.
* Convert each sentence into numerical sequences.

### Step 3: Pad Sequences

* Fix a maximum sentence length.
* Pad shorter sentences and truncate longer ones.

### Step 4: Split Data and Create Batches

* Split dataset into training and testing sets.
* Create PyTorch Dataset and DataLoader for batching.

### Step 5: Build and Train BiLSTM Model

* Create embedding layer, BiLSTM layer, and fully connected layer.
* Train the model using CrossEntropy loss and Adam optimizer.
* Calculate training and validation loss for each epoch.

### Step 6: Test and Predict

* Test the model on unseen sentences.
* Predict entity tags for each word.
* Compare predicted tags with actual tags.

---
## PROGRAM

### Name: Surya Prakash B
### Register Number: 212224230281

```python
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare data
data = pd.read_csv("ner_dataset.csv", encoding="latin1").ffill()
words = list(data["Word"].unique())
tags = list(data["Tag"].unique())

if "ENDPAD" not in words:
    words.append("ENDPAD")

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

data.head(50)

print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

print("Unique tags are:", tags)

# Group words by sentences
class SentenceGetter:
    def __init__(self, data):
        self.grouped = data.groupby("Sentence #", group_keys=False).apply(
            lambda s: [(w, t) for w, t in zip(s["Word"], s["Tag"])]
        )
        self.sentences = list(self.grouped)

getter = SentenceGetter(data)
sentences = getter.sentences


sentences[35]

# Encode sentences
X = [[word2idx[w] for w, t in s] for s in sentences]
y = [[tag2idx[t] for w, t in s] for s in sentences]

word2idx

tag2idx

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

# Pad sequences
max_len = 50
X_pad = pad_sequence([torch.tensor(seq) for seq in X], batch_first=True, padding_value=word2idx["ENDPAD"])
y_pad = pad_sequence([torch.tensor(seq) for seq in y], batch_first=True, padding_value=tag2idx["O"])
X_pad = X_pad[:, :max_len]
y_pad = y_pad[:, :max_len]

X_pad[0]

y_pad[0]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_pad, test_size=0.2, random_state=1)

# Dataset class
class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx],
            "labels": self.y[idx]
        }

train_loader = DataLoader(NERDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(NERDataset(X_test, y_test), batch_size=32)


# Model definition
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, target_size, embedding_dim = 50, hidden_dim = 100):
      super(BiLSTMTagger, self).__init__()
      self.embedding = nn.Embedding(vocab_size, embedding_dim)
      self.dropout = nn.Dropout(0.1)
      self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True, bidirectional = True)
      self.fc = nn.Linear(hidden_dim * 2, target_size)

    def forward(self, x):
      x =self.embedding(x)
      x = self.dropout(x)
      x, _ = self.lstm(x)
      return self.fc(x)


model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses,val_losses=[],[]
    for epoch in range(epochs):
      model.train()
      total_loss=0
      for batch in train_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss)
      model.eval()
      val_loss=0
      with torch.no_grad():
        for batch in test_loader:
          input_ids=batch["input_ids"].to(device)
          labels=batch["labels"].to(device)
          outputs=model(input_ids)
          loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
          val_loss+=loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f},Val Loss={val_loss:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != tag2idx["O"]:
                        true_tags.append(idx2tag[labels[i][j].item()])
                        pred_tags.append(idx2tag[preds[i][j].item()])

# Run training and evaluation
train_losses, val_losses = train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3)
evaluate_model(model, test_loader, X_test, y_test)

# Plot loss
print('Name: Surya Prakash B')
print('Register Number: 212224230281')
history_df = pd.DataFrame({"loss": train_losses, "val_loss": val_losses})
history_df.plot(title="Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Inference and prediction
i = 125
model.eval()
sample = X_test[i].unsqueeze(0).to(device)
output = model(sample)
preds = torch.argmax(output, dim = -1).squeeze().cpu().numpy()
true = y_test[i].numpy()

print('Name: Surya Prakash B')
print('Register Number: 212224230281')
print("{:<15} {:<10} {}\n{}".format("Word", "True", "Pred", "-" * 40))
for w_id, true_tag, pred_tag in zip(X_test[i], y_test[i], preds):
    if w_id.item() != word2idx["ENDPAD"]:
        word = words[w_id.item() - 1]
        true_label = tags[true_tag.item()]
        pred_label = tags[pred_tag]
        print(f"{word:<15} {true_label:<10} {pred_label}")
```

### OUTPUT

## Loss Vs Epoch Plot
<img width="577" height="41" alt="image" src="https://github.com/user-attachments/assets/66ca167f-dcd6-48ad-9b00-077bc4efda02" />
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/9acf2b29-a62d-4264-89f1-4cda3749d666" />

### Sample Text Prediction
<img width="317" height="411" alt="image" src="https://github.com/user-attachments/assets/fea8a96e-e3a8-455d-99b7-f4017fce82e6" />


## Result

The BiLSTM model was successfully trained and was able to predict named entity tags for words with decreasing training and validation loss.

