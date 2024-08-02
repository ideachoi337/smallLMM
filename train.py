from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from dataset import Dataset_C4
from loss import CrossEntropyLoss
from model import Model, ModelArgs
from vocab import VocabInfo
from tokenizer import TokenManager
from tqdm import tqdm
from datetime import datetime

def data_cut(seq, length, save_ratio=0.5, mask_tok = 1):
    B, seq_len = seq.shape
    tri = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(seq.device)
    mask = tri[(length * save_ratio).to(torch.int32)]
    return seq.masked_fill_(mask, mask_tok)
    

def train_one_epoch(args, epoch, model, data_loader, loss_fn, optimizer, device='cuda', writer=None):
    model.train()
    running_loss = 0
    last_loss = 0
    for i, data in tqdm(enumerate(data_loader)):
        text = data['text'].to(device)
        length = data['length'].to(device)
        input_text = data_cut(text, length)

        if (text.size(1)) > args.max_seq_len:
            text = text[:, :args.max_seq_len]
            input_text = input_text[:, :args.max_seq_len]

        optimizer.zero_grad()
        
        outputs = model.train_forward(input_text, text)

        loss = loss_fn(outputs, text)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            if writer is not None:
                writer.add_scalar(f'Loss/train/{epoch}/{i}', last_loss, i)
            running_loss = 0.
            torch.save(
                {
                    "model": "Model",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cost": last_loss,
                    "description": f"Model chkpoint-{epoch}/{i}" ,
                },
                f"./checkpoint-{epoch}-{i}.pt",
            )
    return last_loss

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('./runs/trainer_{}'.format(timestamp))
    epoch_number = 0
    
    EPOCHS = 5
    
    for epoch in range(EPOCHS):
        token_manager = TokenManager()
        args = ModelArgs(vocab_size=len(token_manager.get_vocab().name2val))
        model = Model(args).to('cuda')
        model.train(True)
        optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        loss_fn = CrossEntropyLoss()
        dataset = Dataset_C4(token_manager, train_batch_size=2)
        loss = train_one_epoch(args, epoch, model=model, data_loader=dataset.train_loader, loss_fn=loss_fn, optimizer=optim, writer=writer)
        print(loss) 
        

if __name__ == '__main__':
    main()