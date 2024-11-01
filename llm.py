# LITE LANGUAGE MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import requests
import os
import datetime

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
data = torch.tensor(encode(text), dtype=torch.long)  # (N,)
n = int(0.9*len(data))
train_data = data[:n]  # (train_size,)
val_data = data[n:]    # (val_size,)
dataset = {
    'train': train_data,
    'val': val_data
}

# Get a batch of data
def get_batch(split, dataset):
    data = dataset[split]
    random_indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in random_indices])  # (B, T)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in random_indices])  # (B, T)
    return x, y

# <-------------------Hyperparameters-------------------->
VOCAB_SIZE = len(chars)
N_EMB = 384
BLOCK_SIZE = 256
N_HEAD = 6
DROPOUT = 0.2
N_LAYER = 6
BATCH_SIZE = 64
MAX_ITERS = 5000
EVAL_INTERVAL = 200
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# <------------------------------------------------------>

# layer normalization normalizes across the channels, not across the batch
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim).to(DEVICE)
        self.beta = torch.zeros(dim).to(DEVICE)

    def __call__(self, x):
        xmean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)
        xvar = x.var(dim=-1, keepdim=True)    # (B, T, 1)
        
        out = (x - xmean) / (xvar + self.eps).sqrt()  # (B, T, C)
        out = out * self.gamma + self.beta            # (B, T, C)
        
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # num_heads x (B, T, head_size)
        assert num_heads*head_size == N_EMB
        self.proj = nn.Linear(N_EMB, N_EMB)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * head_size)
        out = self.proj(out)  # (B, T, N_EMB)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMB, head_size, bias=False)
        self.query = nn.Linear(N_EMB, head_size, bias=False)
        self.value = nn.Linear(N_EMB, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))  # (BLOCK_SIZE, BLOCK_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
    
    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = (k @ q.transpose(-2,-1)) / (C ** 0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v  # (B, T, head_size)
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
        return self.net(x)  # (B, T, N_EMB)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # (B, T, N_EMB)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # (B, T, N_EMB)
        x = x + self.ffwd(self.ln2(x))  # (B, T, N_EMB)
        return x
            
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMB)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMB)
        self.blocks = nn.Sequential(*[Block(N_EMB, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = LayerNorm(N_EMB)
        self.lm_head = nn.Linear(N_EMB, VOCAB_SIZE)

    def forward(self, inputs, targets=None):
        _, T = inputs.shape
        token_embeddings = self.token_embedding_table(inputs)  # (B, T, N_EMB)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T, N_EMB)
        x = token_embeddings + position_embeddings  # (B, T, N_EMB)
        x = self.blocks(x)  # (B, T, N_EMB)
        x = self.ln_f(x)  # (B, T, N_EMB)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape           
            logits = logits.view(B*T, C)  # (B*T, VOCAB_SIZE)
            targets = targets.view(B*T)  # (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(inputs[:, -BLOCK_SIZE:])  # (B, T, VOCAB_SIZE)
            logits = logits[:, -1, :]  # (B, VOCAB_SIZE)
            probs = F.softmax(logits, dim=1)  # (B, VOCAB_SIZE)
            char_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            inputs = torch.cat((inputs, char_next), dim=1)  # (B, T+1)
        return inputs

model = LanguageModel().to(DEVICE)
# Calculate and print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {num_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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
            X, Y = get_batch(split, dataset)  # X is (B, T) and Y is (B, T)
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # set the model back to training mode
    return out

for steps in range(MAX_ITERS):
    xb, yb = get_batch('train', dataset)  # (B, T)
    xb = xb.to(DEVICE)
    yb = yb.to(DEVICE)
    logits, loss = model(xb, yb)  # logits: (B, T, VOCAB_SIZE), loss: scalar
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
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # (1, 1)
generated_text = decode(model.generate(context, max_tokens=500)[0].tolist())

# Get current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create filenames with the timestamp
generated_text_filename = f"generated_text_{timestamp}.txt"
model_filename = f"model_weights_{timestamp}.pth"

# Write generated text to file
with open(generated_text_filename, 'w') as f:
    f.write(generated_text)

print(f"Generated text saved to {generated_text_filename}")

# Save the model state dictionary
torch.save(model.state_dict(), model_filename)

print(f"Model weights saved to {model_filename}")
