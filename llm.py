# LITE LANGUAGE MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import requests
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "tinyshakespeare.txt"

if not os.path.exists(filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)
    print(f"Downloaded {filename}")
else:
    print(f"{filename} already exists, skipping download")

with open("tinyshakespeare.txt", "r") as file:
    text = file.read()

print(f"Length of dataset: {len(text)} characters")

# Set the random seed for reproducibility
torch.manual_seed(1337)

# Encoder and decoder
chars = sorted(list(set(text)))
encode = lambda s: [chars.index(c) for c in s]
decode = lambda e: ''.join([chars[i] for i in e])

# Create the dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:] # unbiased estimate of the mnodel performance
dataset = {
    'train': train_data,
    'val': val_data
}

# Get a batch of data
def get_batch(split, dataset):
    data = dataset[split]
    random_indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in random_indices])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in random_indices])
    return x, y

# <-------------------Hyperparameters-------------------->
VOCAB_SIZE = len(chars)
N_EMB = 32
BLOCK_SIZE = 8
BATCH_SIZE = 32
MAX_ITERS =5000
EVAL_INTERVAL = 200
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT = 0.2
N_LAYER = 6
# <------------------------------------------------------>

# layer normalization normalizes across the channels, not across the batch
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim).to(DEVICE)
        self.beta = torch.zeros(dim).to(DEVICE)

    def __call__(self, x):
        xmean = x.mean(dim=-1, keepdim=True)
        xvar = x.var(dim=-1, keepdim=True)
        
        out = (x - xmean) / (xvar + self.eps).sqrt()
        out = out * self.gamma + self.beta
        
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # num_heads x batch x time x head_size
        assert num_heads*head_size == N_EMB
        self.proj = nn.Linear(N_EMB, N_EMB)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # batch x time x num_heads x head_size -> batch x time x (num_heads * head_size)
        out = self.proj(out) # batch x time x N_EMB
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMB, head_size, bias=False)
        self.query = nn.Linear(N_EMB, head_size, bias=False)
        self.value = nn.Linear(N_EMB, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))) # buffer is a non trainable tensor in pytorch jargon
        self.dropout = nn.Dropout(DROPOUT)
    
    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)     # -> B, T, head_size
        q = self.query(x)   # -> B, T, head_size
        v = self.value(x)   # -> B, T, head_size

        wei = (k @ q.transpose(-2,-1)) / (C ** 0.5)  # -> B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # future does not communicate with the past - only a decoder block
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v  # -> B, T, head_size
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # B, T, n_embd
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
            
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMB)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMB)
        self.blocks = nn.Sequential(*[Block(N_EMB, n_head=4) for _ in range(N_LAYER)])
        self.ln_f = LayerNorm(N_EMB)
        self.lm_head = nn.Linear(N_EMB, VOCAB_SIZE)

    def forward(self, inputs, targets=None):
        _, T = inputs.shape
        token_embeddings = self.token_embedding_table(inputs) # (B,T,N_EMB), batch  x time x embedding dimension
        position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,N_EMB), time x embedding dimension
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,VOCAB_SIZE), batch x time x vocab size
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape           
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(inputs[:, -BLOCK_SIZE:]) # crop the input to the last BLOCK_SIZE characters
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=1) # (B, C)
            char_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            inputs = torch.cat((inputs, char_next), dim=1) # (B, T+1)
        return inputs

model = BigramLanguageModel().to(DEVICE)
# Calculate and print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {num_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # takes the gradients and updated the parameters. For a small network we can get away with larger learning rates, typically it would be something like 3-4

start_time = time.time()

@torch.no_grad()
def estimate_loss():
    """
    Estimates the average loss for both training and validation datasets.

    This function:
    1. Sets the model to evaluation mode to disable dropout, etc.
    2. Computes loss for EVAL_INTERVAL batches for both train and val splits.
    3. Calculates the mean loss for each split.
    4. Sets the model back to training mode.
    5. Returns a dictionary with average losses for both splits.

    The purpose is to get a more stable estimate of model performance
    by averaging over multiple batches, helping to track training progress
    and detect overfitting.

    Returns:
        dict: Contains average losses for 'train' and 'val' splits.
    """
    out = {}
    model.eval()  # set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_INTERVAL)
        for k in range(EVAL_INTERVAL):
            X, Y = get_batch(split, dataset)  # X is (B, T, C) and Y is (B, T)
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # set the model back to training mode
    return out

for steps in range(MAX_ITERS):
    xb, yb = get_batch('train', dataset)
    xb = xb.to(DEVICE)
    yb = yb.to(DEVICE)
    logits, loss = model(xb, yb)
    if steps % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end_time = time.time()
training_time = end_time - start_time

print(f"Final loss: {loss.item()}")
print(f"Training time: {training_time:.2f} seconds")

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_tokens=500)[0].tolist()))
