import math
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Hugging Face Datasets
from datasets import load_dataset

# Additional imports for enhancements
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

###############################################################################
# 1. Character-Level Dataset and Vocab
###############################################################################

class CharacterDataset(Dataset):
    """
    Creates a character-level dataset by slicing a single long text string.
    Returns (x, y) pairs where x and y are arrays of integer IDs for chars.
    """
    def __init__(self, text, seq_len, char2idx):
        super().__init__()
        self.text = text
        self.seq_len = seq_len
        self.char2idx = char2idx

    def __len__(self):
        # We'll generate slices up to len(text)-seq_len
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        x_str = self.text[idx : idx + self.seq_len]
        y_str = self.text[idx + 1 : idx + self.seq_len + 1]
        # Convert chars to IDs
        x_ids = [self.char2idx[ch] for ch in x_str]
        y_ids = [self.char2idx[ch] for ch in y_str]
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)

def collate_fn(batch):
    """
    Collate function for DataLoader.
    Stacks (x, y) pairs into (B, T).
    """
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

def build_char_vocab(texts):
    """
    Builds char2idx and idx2char mappings given a list of strings.
    """
    # Collect all characters
    unique_chars = set("".join(texts))
    # Sort for reproducibility
    unique_chars = sorted(list(unique_chars))
    char2idx = {ch: i for i, ch in enumerate(unique_chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char

###############################################################################
# 2. Positional Encoding
###############################################################################

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

###############################################################################
# 3. Small Transformer with Causal Mask
###############################################################################

class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        max_seq_len=512,
        idx2char=None
    ):
        super().__init__()
        self.d_model = d_model
        self.idx2char = idx2char  # For decoding during generation (optional)

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.max_seq_len=max_seq_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head (linear projection to vocab size)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        Returns: logits of shape [batch_size, seq_len, vocab_size].
        """
        # Embedding + position
        x_emb = self.embed_tokens(x) * math.sqrt(self.d_model)
        x_emb = self.pos_encoder(x_emb)

        # Generate a causal mask (subsequent mask) so that position i
        # cannot see positions > i. shape: [seq_len, seq_len]
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        encoded = self.transformer_encoder(x_emb, mask=causal_mask)
        logits = self.fc_out(encoded)  # [batch_size, seq_len, vocab_size]
        return logits

    def generate(
        self, 
        prompt_ids,
        max_length=50, 
        temperature=1.0, 
        top_k=50, 
        top_p=0.95, 
        do_sample=True
    ):
        """
        Simple sampling loop for text generation at char-level.
        Args:
            prompt_ids: A tensor of shape [1, seq_len] with character IDs.
        Returns:
            generated_ids: A tensor of shape [1, seq_len + generated_tokens].
        """
        generated_ids = prompt_ids.clone()

        for _ in range(max_length):
            logits = self.forward(generated_ids[:, -self.max_seq_len:])  # only last seq_len tokens
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Optionally do top-k / top-p sampling
            if do_sample:
                # Top-k
                if top_k > 0:
                    top_k_vals, top_k_idxs = torch.topk(next_token_logits, top_k, dim=-1)
                    mask = next_token_logits < top_k_vals[:, -1].unsqueeze(-1)
                    next_token_logits[mask] = float('-inf')

                # Top-p (nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    next_token_logits[sorted_indices[remove_mask]] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append next token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return generated_ids

    def decode_ids(self, ids):
        """
        Converts a list (or 1D tensor) of character IDs back into a string.
        Requires that self.idx2char is not None.
        """
        if self.idx2char is None:
            raise ValueError("No idx2char mapping found. Provide idx2char to the model for decoding.")
        return "".join([self.idx2char[int(i)] for i in ids])

###############################################################################
# 4. Training & Evaluation
###############################################################################

def train_one_epoch(model, dataloader, optimizer, criterion, device, writer, epoch, scheduler=None, num_iters_generate=None, checkpoint_iters=None, checkpoint_dir=None):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (x, y) in progress_bar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # shape: [batch_size, seq_len, vocab_size]
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
            writer.add_scalar("Learning Rate/Step", scheduler.get_last_lr()[0], epoch * len(dataloader) + batch_idx)

        writer.add_scalar("Loss/Train_iter", loss.item(), epoch * len(dataloader) + batch_idx)

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

        # Print sample generation every N steps
        if num_iters_generate is not None and batch_idx % num_iters_generate == 0:
            # Take the first sample in the batch as a prompt
            prompt = x[:1]  # Shape: [1, seq_len]
            generated = model.generate(prompt, max_length=2*model.max_seq_len, do_sample=False)
            
            # Decode the full generated sequence
            full_generated_ids = generated[0].tolist()
            full_generated_str = model.decode_ids(full_generated_ids)
            
            # Decode the prompt
            prompt_ids = prompt[0].tolist()
            prompt_str = model.decode_ids(prompt_ids)
            
            # Extract the generated continuation
            generated_continuation_str = full_generated_str[len(prompt_str):]
            
            # Print both prompt and generated continuation
            print("\n----- Sample Generation -----")
            print("Prompt:")
            print(repr(prompt_str))
            print("Generated Text:")
            print(repr(generated_continuation_str))
            print("------------------------------\n")

        if checkpoint_iters is not None and batch_idx % checkpoint_iters == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_iter_{batch_idx}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    epoch_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/Train_Epoch", epoch_loss, epoch)
    return epoch_loss


def evaluate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} [Val]")
    with torch.no_grad():
        for batch_idx, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

            writer.add_scalar("Loss/Val_iter", loss.item(), epoch * len(dataloader) + batch_idx)
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

    epoch_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/Val_Epoch", epoch_loss, epoch)
    return epoch_loss

###############################################################################
# 5. Main Script
###############################################################################

def main():
    # ---------------------------
    # 1. Parse Command-Line Arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description="Train a Small Transformer on TinyStories (char-level) with TensorBoard Logging")

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save TensorBoard logs and model checkpoints.')
    parser.add_argument('--experiment_name', type=str, default=f'experiment-{current_time}',
                        help='Name of the experiment for logging purposes.')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length for training.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading.')
    parser.add_argument('--num_train_tokens', type=int, default=None,
                        help='Limit on total chars used from train set (for debugging).')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimensions.')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Num attention heads')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Num transformer layers')
    parser.add_argument('--num_iters_generate', type=int, default=10000,
                        help='Number of iters to print a generation for sanity check')
    parser.add_argument('--checkpoint_iters', type=int, default=10000,
                        help='Number of iters to save a model checkpoint')

    args = parser.parse_args()

    # ---------------------------
    # 2. Setup Output Directories
    # ---------------------------
    log_dir = os.path.join(args.output_dir, args.experiment_name, 'logs')
    checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---------------------------
    # 3. Initialize TensorBoard
    # ---------------------------
    writer = SummaryWriter(log_dir=log_dir)

    # ---------------------------
    # 4. Device Configuration
    # ---------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ---------------------------
    # 5. Load TinyStories Data
    # ---------------------------
    # TinyStories has 'train' and 'validation' splits
    train_data = load_dataset("roneneldan/TinyStories", split="train")
    valid_data = load_dataset("roneneldan/TinyStories", split="validation")

    # Each item is typically {'text': 'A short story...'}.
    # Collect them into lists of strings.
    train_texts = [item['text'] for item in train_data]
    valid_texts = [item['text'] for item in valid_data]
    print("Loaded training and validation data!")

    # Build vocabulary over all data (train + valid).
    # Or you can build only on train_texts if you want.
    char2idx, idx2char = build_char_vocab(train_texts + valid_texts)
    print(f"Built vocabulary with {len(char2idx)} characters.")

    # Flatten into a single text string (with newline separators) for each split
    train_text = "\n".join(train_texts)
    valid_text = "\n".join(valid_texts)

    # Optionally truncate training data for debugging
    if args.num_train_tokens is not None:
        train_text = train_text[:args.num_train_tokens]

    # 2.1 Print and log total characters in training set
    num_train_chars = len(train_text)
    print(f"Number of chars in training set: {num_train_chars}")
    writer.add_scalar("Data/Number of Training Chars", num_train_chars, 0)

    # ---------------------------
    # 6. Create Dataset & DataLoaders
    # ---------------------------
    train_dataset = CharacterDataset(train_text, seq_len=args.seq_len, char2idx=char2idx)
    val_dataset = CharacterDataset(valid_text, seq_len=args.seq_len, char2idx=char2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda")
    )

    # ---------------------------
    # 7. Initialize Model
    # ---------------------------
    vocab_size = len(char2idx)
    model = SmallTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.n_layers,
        dropout=0.1,
        max_seq_len=args.seq_len,
        idx2char=idx2char   # so we can decode for debugging
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params}")
    print(f"Trainable model parameters: {trainable_params}")
    writer.add_scalar("Model/Total Parameters", total_params, 0)
    writer.add_scalar("Model/Trainable Parameters", trainable_params, 0)

    # ---------------------------
    # 8. Loss, Optimizer, Scheduler
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr/10)

    # ---------------------------
    # 9. Training Loop
    # ---------------------------
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, writer, epoch, scheduler, num_iters_generate=args.num_iters_generate, checkpoint_iters=args.checkpoint_iters, checkpoint_dir=checkpoint_dir)
        val_loss = evaluate(model, val_loader, criterion, DEVICE, writer, epoch)

        end_time = time.time()
        epoch_duration = end_time - start_time
        eta = epoch_duration * (args.epochs - epoch - 1)

        writer.add_scalar("Time/Epoch", epoch_duration, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Epoch Time: {epoch_duration:.2f}s | ETA: {eta/60:.2f}m")

    # ---------------------------
    # 10. Close Writer
    # ---------------------------
    writer.close()

if __name__ == "__main__":
    main()
