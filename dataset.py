from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tokenizer import TokenManager

class Dataset_C4:
    def __init__(self, token_manager: TokenManager, train_batch_size: int = 16, valid_batch_size: int = 16, streaming=True):
        self.token_manager = token_manager
        self.pad_id = self.token_manager.get_pad_id()
        self.train_dataset = load_dataset("allenai/c4", "en", split='train', streaming=streaming).map(self.tokenize).with_format('torch')
        self.valid_dataset = load_dataset("allenai/c4", "en", split='validation', streaming=streaming).map(self.tokenize).with_format('torch')
        self.train_loader = DataLoader(self.train_dataset, batch_size = train_batch_size, collate_fn=self.collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size = train_batch_size, collate_fn=self.collate_fn)

    def tokenize(self, seq):
        return {'text': self.token_manager.tokenize_text(seq['text'])}
    
    def collate_fn(self, samples):
        max_len = max([len(sample['text']) for sample in samples]) 
        collate = []
        length = []
        for sample in samples:
            diff = max_len - len(sample['text'])
            if diff > 0:
                pad = torch.ones(size=(diff,), dtype=torch.int) * self.pad_id
                collate.append(torch.cat([sample['text'], pad], dim=0))
            else:
                collate.append(sample['text'])
            length.append(len(sample['text']))
        return {'text': torch.stack(collate), 'length': torch.Tensor(length)}