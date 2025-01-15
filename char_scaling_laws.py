import math
import time
import os
import argparse
import csv 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from torch.utils.data import Sampler
import json

class TokenLimitedSampler(Sampler):
    def __init__(self, data_source, num_tokens):
        self.data_source = data_source
        self.num_tokens = num_tokens

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source))[:self.num_tokens].tolist())

    def __len__(self):
        return self.num_tokens


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
        x_ids = [self.char2idx.get(ch, self.char2idx[' ']) for ch in x_str]  # Handle unknown chars
        y_ids = [self.char2idx.get(ch, self.char2idx[' ']) for ch in y_str]
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
    # Ensure a padding character exists
    unique_chars.add(' ')  # Adding space as padding if not present
    # Sort for reproducibility
    unique_chars = sorted(list(unique_chars))
    char2idx = {ch: i for i, ch in enumerate(unique_chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char


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
        self.max_seq_len = max_seq_len  # Updated to store max_seq_len

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

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


def gaussian_kernel(size=10, sigma=2.0):
    """
    Create a 1D Gaussian kernel of length `size` and standard deviation `sigma`.
    This kernel will be normalized to sum to 1.
    """
    # E.g. for size=10, this creates points from -4.5 to +4.5
    x = np.linspace(-(size-1)/2., (size-1)/2., size)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    return kernel

def add_to_log_dict(log_dict: dict | None, key, y_val, x_val):
    if log_dict is None:
        return
    if key not in log_dict:
        log_dict[key] = []
    log_dict[key].append(
        {
            "value": y_val,
            "step": x_val,
        }
    )

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    writer,
    epoch,
    scheduler=None, 
    num_iters_generate=None,
    checkpoint_iters=None,
    checkpoint_dir=None, 
    window_size=10,  # Added a window_size for clarity
    num_params: int | None = None,
    log_dict: dict | None = None,
    log_dir: str | None = None,
):
    model.train()
    
    # Initialize total loss
    total_loss = 0.0
    
    # We'll keep track of recent batch losses in this list:
    recent_losses = []
    # Pre-compute a Gaussian kernel of desired size:
    kernel = gaussian_kernel(size=window_size, sigma=2.0)
    
    # Record the start time of the epoch
    epoch_start_time = time.time()
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (x, y) in progress_bar:
        batch_start_time = time.time()  # Start time of the current batch
        
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # shape: [batch_size, seq_len, vocab_size]
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        # ----------------------------
        # 1. Gradient Clipping
        # ----------------------------
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("Learning Rate/Step", current_lr, epoch * len(dataloader) + batch_idx)

        # Log the raw loss each iteration
        writer.add_scalar("Loss/Train_iter", loss.item(), epoch * len(dataloader) + batch_idx)


        # ----------------------------
        # 2. Maintain a rolling buffer
        # ----------------------------
        recent_losses.append(loss.item())
        if len(recent_losses) > window_size:
            # Keep only the most recent `window_size` losses
            recent_losses.pop(0)

        # ----------------------------
        # 3. Compute Gaussian-smoothed loss
        # ----------------------------
        current_window_length = len(recent_losses)
        # Take only the last `current_window_length` points of the kernel
        # so that it matches how many recent losses we have.
        used_kernel = kernel[-current_window_length:]
        # Renormalize in case we sliced the kernel
        used_kernel = used_kernel / used_kernel.sum()
        
        # Weighted sum of recent_losses by the (sliced) Gaussian kernel
        smoothed_loss = sum(l * k for l, k in zip(recent_losses, used_kernel))

        # ----------------------------
        # 4. Calculate and Log ETA
        # ----------------------------
        elapsed_time = time.time() - epoch_start_time  # Total elapsed time since epoch start
        avg_time_per_batch = elapsed_time / (batch_idx + 1)  # Average time per batch
        remaining_batches = len(dataloader) - (batch_idx + 1)  # Batches left
        eta_seconds = avg_time_per_batch * remaining_batches  # Estimated remaining time in seconds

        # Convert ETA to a more readable format (e.g., minutes and seconds)
        eta_minutes = int(eta_seconds // 60)
        eta_secs = int(eta_seconds % 60)
        # eta_formatted = f"{eta_minutes}m {eta_secs}s"

        # Log ETA to TensorBoard
        writer.add_scalar("Time/ETA", eta_seconds, epoch * len(dataloader) + batch_idx)

        # ----------------------------
        # 5. Update TQDM progress bar
        # ----------------------------
        progress_bar.set_postfix({
            "Loss (raw)": f"{loss.item():.4f}",
            "Smoothed Loss": f"{smoothed_loss:.4f}",
        })

        # ----------------------------
        # 6. Log smoothed loss to TensorBoard
        # ----------------------------
        writer.add_scalar("Loss/Train_iter_smoothed", smoothed_loss, 
                          epoch * len(dataloader) + batch_idx)
        add_to_log_dict(log_dict, "Loss/Train_iter_smoothed", smoothed_loss, epoch * len(dataloader) + batch_idx)

        if num_params is not None:
            batch_size = x.size(0)
            seq_len = x.size(1)
            num_iters = epoch * len(dataloader) + (batch_idx + 1)
            num_tokens = batch_size * seq_len * num_iters
            approx_num_flops = 6 * num_params * num_tokens
            writer.add_scalar("Loss/Train_per_flops", loss.item(), approx_num_flops)
            writer.add_scalar("Loss/Train_per_flops_smoothed", smoothed_loss, approx_num_flops)

        # ----------------------------
        # 7. Generate and Log Sample Text
        # ----------------------------
        if (batch_idx == len(dataloader) - 1) or (num_iters_generate is not None and (batch_idx % num_iters_generate == 0)):
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
            
            print("\n----- Sample Generation -----")
            print("Prompt:")
            print(repr(prompt_str))
            print("Generated Text:")
            print(repr(generated_continuation_str))
            print("------------------------------\n")

            text_out = "\nSample Prompt:\n" + repr(prompt_str) + "\nGenerated Text:\n" + repr(generated_continuation_str)
            writer.add_text(
                "Generated Text",
                text_out,
                global_step=epoch * len(dataloader) + batch_idx
            )

        # ----------------------------
        # 8. Save Checkpoints Periodically
        # ----------------------------
        if checkpoint_iters is not None and checkpoint_dir and (batch_idx % checkpoint_iters == 0):
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_iter_{batch_idx+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        if log_dict is not None and batch_idx % 100 == 0:
            assert log_dir is not None
            log_dir_tmp = os.path.join(log_dir, "log_file.json")
            with open(log_dir_tmp, "w") as f:
                json.dump(log_dict, f, indent=4)

    if log_dict is not None:
        assert log_dir is not None
        log_dir_tmp = os.path.join(log_dir, "log_file.json")
        with open(log_dir_tmp, "w") as f:
            json.dump(log_dict, f, indent=4)
    # Return the epoch average smoothed loss
    return smoothed_loss



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

def main():
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
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate for the optimizer.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading.')
    parser.add_argument('--num_train_tokens', type=int, default=None,
                        help='Limit on total chars used from train set (for debugging).')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimensions.')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads.')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of transformer layers.')
    parser.add_argument('--num_iters_generate', type=int, default=5000,
                        help='Number of iters to print a generation for sanity check.')
    parser.add_argument('--checkpoint_iters', type=int, default=5000,
                        help='Number of iters to save a model checkpoint.')
    parser.add_argument('--run_val', type=bool, default=False,
                        help='Whether or not to run validation after each epoch.')

    args = parser.parse_args()

    log_dict = {}

    log_dir = os.path.join(args.output_dir, args.experiment_name, 'logs')
    checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    args_dict = vars(args)
    args_json_path = os.path.join(args.output_dir, args.experiment_name, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Saved arguments to {args_json_path}")

    writer = SummaryWriter(log_dir=log_dir)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    train_data = load_dataset("roneneldan/TinyStories", split="train")
    valid_data = load_dataset("roneneldan/TinyStories", split="validation")

    train_texts = [item['text'] for item in train_data]
    valid_texts = [item['text'] for item in valid_data]
    print("Loaded training and validation data!")

    char2idx, idx2char = build_char_vocab(train_texts + valid_texts)
    print(f"Built vocabulary with {len(char2idx)} characters.")

    train_text = "\n".join(train_texts)
    valid_text = "\n".join(valid_texts)

    num_train_chars = len(train_text)
    print(f"Number of chars in training set: {num_train_chars}")
    writer.add_scalar("Data/Number of Training Chars", num_train_chars, 0)
    add_to_log_dict(log_dict, "Data/Number of Training Chars", num_train_chars, 0)

    train_dataset = CharacterDataset(train_text, seq_len=args.seq_len, char2idx=char2idx)
    val_dataset = CharacterDataset(valid_text, seq_len=args.seq_len, char2idx=char2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True if args.num_train_tokens is None else False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
        sampler=TokenLimitedSampler(train_dataset, int(round(args.num_train_tokens / args.seq_len))) if args.num_train_tokens is not None else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda")
    )

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

    add_to_log_dict(log_dict, "Model/Total Parameters", total_params, 0)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )  # from https://wandb.ai/vincenttu/blog_posts/reports/Meta-AI-Released-LLaMA--VmlldzozNjM5MTAz

    total_steps = args.epochs * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.1*args.lr)

    print(f"Num training iterations: {len(train_loader)}")
    num_training_tokens = len(train_loader) * args.seq_len * args.batch_size
    print(f"Num training tokens: {num_training_tokens}")
    assert num_training_tokens <= len(train_text), "More training tokens than available in the dataset!"
    print(f"Estimated Petaflops: {6 * total_params * num_training_tokens / 1e15}")


    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            writer,
            epoch,
            scheduler,
            num_iters_generate=args.num_iters_generate,
            checkpoint_iters=args.checkpoint_iters,
            checkpoint_dir=checkpoint_dir,
            num_params=total_params,
            log_dict=log_dict,
            log_dir=log_dir,
        )
        if args.run_val:
            val_loss = evaluate(model, val_loader, criterion, DEVICE, writer, epoch)

            end_time = time.time()
            epoch_duration = end_time - start_time
            eta = epoch_duration * (args.epochs - epoch - 1)

            writer.add_scalar("Time/Epoch", epoch_duration, epoch)
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                f"| Epoch Time: {epoch_duration:.2f}s | ETA: {eta/60:.2f}m"
            )
        else:
            end_time = time.time()
            epoch_duration = end_time - start_time
            eta = epoch_duration * (args.epochs - epoch - 1)

            writer.add_scalar("Time/Epoch", epoch_duration, epoch)
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} "
                f"| Epoch Time: {epoch_duration:.2f}s | ETA: {eta/60:.2f}m"
            )

    writer.close()

if __name__ == "__main__":
    main()
