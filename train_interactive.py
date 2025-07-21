import os
import torch
import tiktoken
from model import GPTConfig, GPT

# Settings (adjust as needed)
block_size = 64
n_layer = 2
n_head = 2
n_embd = 64
learning_rate = 3e-4
checkpoint_path = 'out-interactive/ckpt.pt'

# Create output directory if needed
os.makedirs('out-interactive', exist_ok=True)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

def get_model(vocab_size):
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config)
    return model

# Try to load existing model
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = get_model(checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Loaded existing model.')
else:
    # Use GPT-2 vocab size
    vocab_size = enc.n_vocab
    model = get_model(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print('Initialized new model.')

model.train()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print('Type your text. Each line will be used to train the model. Press Ctrl+C to exit.')

try:
    while True:
        line = input('>>> ')
        ids = enc.encode_ordinary(line)
        ids.append(enc.eot_token)
        if len(ids) < 2:
            continue
        # Pad or trim to block_size
        if len(ids) < block_size:
            ids = [enc.eot_token] * (block_size - len(ids)) + ids
        else:
            ids = ids[-block_size:]
        x = torch.tensor([ids[:-1]], dtype=torch.long).to(device)
        y = torch.tensor([ids[1:]], dtype=torch.long).to(device)
        optimizer.zero_grad()
        loss, _ = model(x, y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        print(f'Trained on: "{line}" | Loss: {loss.item():.4f}')
        # Save checkpoint after each line
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': enc.n_vocab,
            'model_args': {
                'block_size': block_size,
                'vocab_size': enc.n_vocab,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
            },
            # for compatibility with sample.py
            'model': model.state_dict(),
        }, checkpoint_path)
        print(f'Model saved to {checkpoint_path}')
except KeyboardInterrupt:
    print('\nExiting interactive training.')
