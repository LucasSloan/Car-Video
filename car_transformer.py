import math
import os
from tempfile import TemporaryDirectory
import time

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader

from datasets import load_dataset
import numpy as np

BOS_TOKEN = 1024
TOKENS_PER_FRAME = 129
BS = 10
CONTEXT_SIZE_FRAMES = 20
N_FRAMES = 1200
N = N_FRAMES - 20
N_TOKENS = 1025 # size of vocabulary
EM_SIZE = 200 # embedding dimension
D_HID = 200 # dimension of the feedforward network model in ``nn.TransformerEncoder``
N_LAYERS = 2 # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
N_HEAD = 2 # number of heads in ``nn.MultiheadAttention``
DROPOUT = 0.2 # dropout probability
LR = 5.0 # learning rate
EPOCHS = 3

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
    
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

        indices = np.arange(0, N*TOKENS_PER_FRAME)
        indices = np.array(np.split(indices, N//CONTEXT_SIZE_FRAMES))
        self.indices = indices


    def __len__(self):
        return len(self.files) * 59

    def __getitem__(self, index):
        file_index = index // 59
        slice_index = index % 59

        tokens = np.load(self.files[file_index])
        tokens = tokens.reshape(N_FRAMES, TOKENS_PER_FRAME-1) # TOKENS_PER_FRAME includes the BOS token
        tokens = np.c_[np.ones(len(tokens), dtype=np.int64)*BOS_TOKEN, tokens]
        tokens = tokens.reshape(-1)

        ii = self.indices[slice_index]

        x = tokens[ii]
        y = tokens[ii+1]

        return x, y


def train(model: nn.Module, device, train_dl, criterion, optimizer, scheduler, epoch) -> None:
    model.train() # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_dl)
    for batch, data in enumerate(train_dl):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            # what shape is the output normally?
            output_flat = preds.view(-1, N_TOKENS)
            loss = criterion(output_flat, y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor, criterion) -> float:
    model.eval() # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for data in eval_data:
            x, y = data
            seq_len = x.size(0)
            preds = model(x)
            preds_flat = preds.view(-1, N_TOKENS)
            total_loss += seq_len * criterion(preds_flat, y).item()
    return total_loss / (len(eval_data) - 1)

def main():
    ds = load_dataset("commaai/commavq", num_proc=8)

    train_files = []
    for i in range(40):
        train_files.extend(ds[str(i)]['path'])

    train_ds = Dataset(train_files)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, drop_last=True)
    val_ds = Dataset(ds['40']['path'])
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=True, num_workers=8, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(N_TOKENS, EM_SIZE, N_HEAD, D_HID, N_LAYERS, DROPOUT).to(device)
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train(model, device, train_dl, criterion, optimizer, scheduler, epoch)
            val_loss = evaluate(model, val_dl, criterion)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()

if __name__ == "__main__":
    main()