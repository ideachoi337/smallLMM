from tokenizers import Tokenizer
from vocab import VocabInfo, VocabTranslation
from image_tokenizer import ImageTokenizer
from PIL import Image
import torch
import json

class TokenManager:
    def __init__(
        self,
        tokenizer_path = '../tokenizer/text_tokenizer.json',
        vqgan_img_path = '../tokenizer/vqgan.ckpt',
        vqgan_cfg_path = '../tokenizer/vqgan.yaml',
        device = 'cuda',
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab = VocabInfo(json.load(open(tokenizer_path))['model']['vocab'])
        self.translation = VocabTranslation(self.vocab, device=device)
        self.image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg_path, ckpt_path=vqgan_img_path, device=device
        )

    def tokenize(self, seq):
        token = []
        token.append(self.vocab.begin_sequence)
        for s in seq:
            if s['type'] == 'text':
                token += self.tokenizer.encode(s['content']).ids
            elif s['type'] == 'image':
                img = Image.open(s['content'])
                token.append(self.vocab.begin_image)
                token += self.translation.convert_img2bp2(self.image_tokenizer.img_tokens_from_pil(img)).tolist()
                token.append(self.vocab.end_image)
        token.append(self.vocab.end_sequence)
        return torch.tensor(token)

    def tokenize_text(self, seq):
        token= []
        token.append(self.vocab.begin_sequence)
        token += self.tokenizer.encode(seq).ids
        token.append(self.vocab.end_sequence)
        return torch.tensor(token)

    def get_pad_id(self):
        return self.vocab.pad_id
       
    def get_vocab(self):
        return self.vocab 