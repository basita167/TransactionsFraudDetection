import pandas as pd
import torch
import torch.nn as nn
import math

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# =========================
# 1. LOAD + PREPROCESS DATA
# =========================
df = pd.read_csv("/kaggle/input/datasets/abdulbasit1287/transactionslegalityds1-0/trasactionsDataSet1.0.csv")

# Encode addresses
le = LabelEncoder()
all_addresses = list(df['from_address']) + list(df['to_address'])
le.fit(all_addresses)

df['from_id'] = le.transform(df['from_address'])
df['to_id'] = le.transform(df['to_address'])

# Tabular features
tabular_cols = [
    'amount_usd',
    'gas_fee',
    'account_age_days',
    'num_prev_transactions',
    'is_flagged_mixer',
    'is_suspicious_pattern'
]

scaler = StandardScaler()
df[tabular_cols] = scaler.fit_transform(df[tabular_cols])

# Inputs
X_addr = df[['from_id', 'to_id']].values
X_tab = df[tabular_cols].values
y = df['legal_or_fraud'].values

# Train-test split
X_addr_train, X_addr_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
    X_addr, X_tab, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_addr_train = torch.tensor(X_addr_train, dtype=torch.long)
X_tab_train = torch.tensor(X_tab_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_addr_test = torch.tensor(X_addr_test, dtype=torch.long)
X_tab_test = torch.tensor(X_tab_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# =========================
# 2. TRANSFORMER FROM SCRATCH
# =========================

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)

        return torch.matmul(weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention(embed_dim) for _ in range(num_heads)
        ])
        self.fc = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.fc(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =========================
# 3. HYBRID MODEL
# =========================

class HybridModel(nn.Module):
    def __init__(self, num_addresses, tabular_dim,
                 embed_dim=64, num_heads=2, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(num_addresses, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers)

        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, addr_input, tab_input):
        x_addr = self.embedding(addr_input)   # (batch, 2, embed_dim)
        x_addr = self.encoder(x_addr)
        x_addr = x_addr.mean(dim=1)

        x_tab = self.tabular_net(tab_input)

        x = torch.cat([x_addr, x_tab], dim=1)
        return self.classifier(x)


# =========================
# 4. TRAINING
# =========================

model = HybridModel(
    num_addresses=len(le.classes_),
    tabular_dim=len(tabular_cols)
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_addr_train, X_tab_train).squeeze()

    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# =========================
# 5. EVALUATION
# =========================

model.eval()

with torch.no_grad():
    preds = model(X_addr_test, X_tab_test).squeeze()
    preds = (preds > 0.5).float()

accuracy = (preds == y_test).float().mean()
print("Test Accuracy:", accuracy.item())
