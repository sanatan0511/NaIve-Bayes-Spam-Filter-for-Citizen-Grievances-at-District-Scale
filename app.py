import streamlit as st
import pandas as pd
import numpy as np
import spacy
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torch.utils.data import TensorDataset, DataLoader

st.set_page_config(
    page_title="Spam Detection NLP + DL + LLM",
    layout="wide"
)

st.title("📨 Advanced SMS Spam Detection System")
st.caption("NLP + ML + Deep Learning + Transformer | Streamlit App")

nltk.download("vader_lexicon")
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_data
def load_data():
    df = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        names=["label", "text"]
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

data = load_data()


def preprocess(text):
    doc = nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    )

data["clean_text"] = data["text"].apply(preprocess)

data["sentiment"] = data["text"].apply(
    lambda x: sid.polarity_scores(x)["compound"]
)


st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Messages", len(data))
with col2:
    st.metric("Spam Messages", int(data["label"].sum()))

st.dataframe(data.head())


st.header("📈 Exploratory Data Analysis")

fig, ax = plt.subplots()
sns.countplot(x=data["label"], ax=ax)
ax.set_xticklabels(["Ham", "Spam"])
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.histplot(data["sentiment"], bins=30, kde=True, ax=ax2)
ax2.set_title("Sentiment Distribution")
st.pyplot(fig2)


st.header("🤖 Train Machine Learning Model")

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data["clean_text"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb = MultinomialNB()
nb.fit(X_train, y_train)

pred = nb.predict(X_test)
acc = accuracy_score(y_test, pred)

st.success(f"Naive Bayes Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)


st.header("🧠 Deep Learning Model (BiLSTM)")

def build_vocab(texts, max_vocab=8000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {w: i+2 for i, (w, _) in enumerate(counter.most_common(max_vocab))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(data["clean_text"])

def encode(text, max_len=50):
    ids = [vocab.get(w, 1) for w in text.split()][:max_len]
    return ids + [0] * (max_len - len(ids))

X_seq = np.array([encode(t) for t in data["clean_text"]])

class BiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

model = BiLSTM(len(vocab)).to(device)

X_tensor = torch.tensor(X_seq, dtype=torch.long)
y_tensor = torch.tensor(data["label"].values, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

st.info("Training BiLSTM (3 epochs, demo training)")

for epoch in range(3):
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        print(Xb,yb)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    st.write(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

st.info("BiLSTM architecture loaded (training skipped for speed)")


st.header("✍️ Sentence Prediction")

user_text = st.text_area("Enter an SMS message:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = preprocess(user_text)
        vec = tfidf.transform([clean])
        pred = nb.predict(vec)[0]
        prob = nb.predict_proba(vec)[0][pred]

        if pred == 1:
            st.error(f"x SPAM ({prob*100:.2f}%)")
        else:
            st.success(f"✅ HAM ({prob*100:.2f}%)")

        st.write("Sentiment Score:", sid.polarity_scores(user_text)["compound"])

st.header("LLM-style:)")

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        encoder = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embed(x).permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)

transformer_model = MiniTransformer(len(vocab)).to(device)