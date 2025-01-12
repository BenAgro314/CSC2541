import math
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Hugging Face Datasets components
from datasets import load_dataset
from transformers import AutoTokenizer

# Additional imports for enhancements
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

###############################################################################
# 1. Data Preparation
###############################################################################

def chunkify(data_ids, seq_len):
    """
    Splits the entire dataset of token IDs into (input, target) chunks of size seq_len.
    Returns a list of tuples (x, y).
    """
    chunks = []
    for i in range(0, len(data_ids) - seq_len):
        x = data_ids[i : i + seq_len]
        y = data_ids[i + 1 : i + seq_len + 1]
        chunks.append((x, y))
    return chunks

class WikiText2Dataset(Dataset):
    """
    Creates a dataset of (x, y) pairs from WikiText-2 token IDs,
    chunked to a fixed sequence length.
    """
    def __init__(self, data_ids, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.samples = chunkify(data_ids, seq_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

def collate_fn(batch):
    """
    Collate function for DataLoader.
    Takes a list of (x, y) pairs and stacks them into (B, T).
    """
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        max_seq_len=512
    ):
        super().__init__()
        self.d_model = d_model
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

        # Transformer Encoder
        encoded = self.transformer_encoder(x_emb, mask=causal_mask)

        # Project to vocab
        logits = self.fc_out(encoded)
        return logits

###############################################################################
# 4. Training & Evaluation
###############################################################################

def train_one_epoch(model, dataloader, optimizer, criterion, device, writer, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (x, y) in progress_bar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)          # shape: [batch_size, seq_len, vocab_size]
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        # Update the learning rate scheduler if provided
        if scheduler:
            scheduler.step()
            writer.add_scalar("Learning Rate/Step", scheduler.get_last_lr()[0], epoch * len(dataloader) + batch_idx)

        # Log training loss per iteration
        writer.add_scalar("Loss/Train_iter", loss.item(), epoch * len(dataloader) + batch_idx)

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

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

            # Log validation loss per iteration
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
    parser = argparse.ArgumentParser(description="Train a Small Transformer on WikiText-2 with TensorBoard Logging")

    # get a nice parsed version of date and time
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save TensorBoard logs and model checkpoints.')
    parser.add_argument('--experiment_name', type=str, default=f'experiment-{current_time}',
                        help='Name of the experiment for logging purposes.')
    parser.add_argument('--seq_len', type=int, default=64,
                        help='Sequence length for training.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading.')
    parser.add_argument('--dataset', type=str, default="wikitext-103-raw-v1",
                        help='Number of worker threads for data loading.') # "wikitext-2-raw-v1"

    args = parser.parse_args()

    # ---------------------------
    # 2. Setup Output Directories
    # ---------------------------
    log_dir = os.path.join(args.output_dir, args.experiment_name, 'logs')
    checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---------------------------
    # 3. Initialize TensorBoard SummaryWriter
    # ---------------------------
    writer = SummaryWriter(log_dir=log_dir)

    # ---------------------------
    # 4. Device Configuration
    # ---------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ---------------------------
    # 5. Load and Prepare Data
    # ---------------------------
    # 1) Load raw WikiText-2 data using Hugging Face Datasets
    dataset = load_dataset("wikitext", args.dataset)
    train_data = dataset["train"]["text"]
    valid_data = dataset["validation"]["text"]

    # 2) Initialize tokenizer (use a basic tokenizer or any other)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add pad token if necessary

    # 3) Tokenize and flatten the data with truncation to prevent long sequences
    def tokenize_and_flatten(texts, tokenizer, max_length=512):
        tokens = []
        for line in texts:
            if line.strip() == "":
                continue  # Skip empty lines
            encoded = tokenizer.encode(
                line,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True
            )
            tokens.extend(encoded)
        return tokens

    train_ids = tokenize_and_flatten(train_data, tokenizer, max_length=512)
    valid_ids = tokenize_and_flatten(valid_data, tokenizer, max_length=512)

    # 2.1 **Print and Log Number of Tokens in Training Set**
    num_train_tokens = len(train_ids)
    print(f"Number of tokens in the training set: {num_train_tokens}")
    writer.add_scalar("Data/Number of Training Tokens", num_train_tokens, 0)  # Step 0

    # 4) Create datasets for train and validation
    train_dataset = WikiText2Dataset(train_ids, seq_len=args.seq_len)
    val_dataset = WikiText2Dataset(valid_ids, seq_len=args.seq_len)

    # 5) Create dataloaders with tqdm and pin_memory for faster data transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE == "cuda" else False
    )

    # ---------------------------
    # 6. Initialize Model
    # ---------------------------
    model = SmallTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,       # adjust as needed
        n_heads=4,         # adjust as needed
        num_layers=2,      # adjust as needed
        dropout=0.1,
        max_seq_len=args.seq_len
    ).to(DEVICE)

    # 1.1 **Print and Log Model Parameters**
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params}")
    print(f"Trainable model parameters: {trainable_params}")

    # Log to TensorBoard
    writer.add_scalar("Model/Total Parameters", total_params, 0)          # Step 0
    writer.add_scalar("Model/Trainable Parameters", trainable_params, 0)  # Step 0

    # ---------------------------
    # 7. Define Loss and Optimizer
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------
    # 8. Implement Cosine Learning Rate Scheduler
    # ---------------------------
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Log initial learning rate
    # initial_lr = optimizer.param_groups[0]['lr']
    # writer.add_scalar("Learning Rate/Initial", initial_lr, 0)  # Step 0

    # ---------------------------
    # 9. Training Loop
    # ---------------------------
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training Phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, writer, epoch, scheduler)
        
        # Validation Phase
        val_loss = evaluate(model, val_loader, criterion, DEVICE, writer, epoch)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        eta = epoch_duration * (args.epochs - epoch - 1)

        # Log epoch duration to TensorBoard
        writer.add_scalar("Time/Epoch", epoch_duration, epoch)

        # Print Epoch Summary with Estimated Time to Finish
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Epoch Time: {epoch_duration:.2f}s | ETA: {eta/60:.2f}m")

        # Save Model Checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    # ---------------------------
    # 10. Close the TensorBoard writer
    # ---------------------------
    writer.close()

if __name__ == "__main__":
    main()
