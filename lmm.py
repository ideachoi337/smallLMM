import base64
import io
import json
import math
import queue
import threading
from dataclasses import dataclass, field
from tqdm import tqdm
from enum import Enum
from multiprocessing import managers, queues, synchronize
from typing import Literal, Union

import PIL
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL.Image import Image
from tokenizers import Tokenizer

from image_tokenizer import ImageTokenizer
from model import Transformer
from vocab import VocabInfo, VocabTranslation

@dataclass
class Options:
    @dataclass
    class Text:
        repetition_penalty: float = 1.2
        temp: float = 1.0
        top_p: float = 0.9
        greedy: bool = False

    @dataclass
    class Image:
        @dataclass
        class CFG:
            guidance_scale_text: float = 3.0
            guidance_scale_image: float = 1.2

        cfg: CFG = field(default_factory=CFG)
        temp: float = 0.7
        top_p: float = 0.9
        greedy: bool = False

    max_seq_len: int = 4096
    max_gen_len: int = 4096
    seed: int | None = None
    txt: Text | bool = True
    img: Image | bool = True
    extra_eos_tokens: list[int | str] = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.txt is True:
            self.txt = Options.Text()
        if self.img is True:
            self.img = Options.Image()
            
class TokenManager:
    def __init__(
        self,
        tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str,
        device: str | None = None,
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab = VocabInfo(json.load(open(tokenizer_path))["model"]["vocab"])
        self.translation = VocabTranslation(self.vocab, device=device)
        self.image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device=device
        )

    def pil_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> PIL.Image:
        image_tensor = self.translation.convert_bpe2img(bpe_tokens)
        if image_tensor.shape[0] < 1024:
            padding = (
                torch.ones(
                    [1024 - image_tensor.shape[0]],
                    dtype=int,
                    device=image_tensor.device,
                )
                * image_tensor[0]
            )
            image_tensor = torch.cat((image_tensor, padding)).unsqueeze(0)

        return self.image_tokenizer.pil_from_img_toks(image_tensor)

    def png_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> bytes:
        pil = self.pil_from_bpe_tokens(bpe_tokens)
        img_io = io.BytesIO()
        pil.save(img_io, format="PNG")
        return img_io.getvalue()

    def tokenize_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def tokenize_image(self, img: Image) -> list[int]:
        return (
            [self.vocab.begin_image]
            + self.translation.convert_img2bp2(
                self.image_tokenizer.img_tokens_from_pil(img)   # [0 : 8191], vqgan codebook ids
            ).tolist()
            + [self.vocab.end_image]
        )

    def tokenize_b64img(self, b64img: str) -> list[int]:
        image_data = base64.b64decode(b64img)
        image_file = io.BytesIO(image_data)
        return self.tokenize_image(PIL.Image.open(image_file))

    def tokens_from_ui(self, inputs: list[dict]) -> list[int]:
        tokens = [self.vocab.bos_id]
        for input_ in inputs:
            if input_["type"] == "text":
                tokens += self.tokenize_text(input_["value"])
            elif input_["type"] == "image":
                if isinstance(input_["value"], str):
                    if input_["value"].startswith("data:"):
                        # Value Format: 'data:image/[^;]+;base64,[A-Za-z0-9+/]+={0,2}'
                        tokens += self.tokenize_b64img(input_["value"].split(",", 1)[1])
                    elif input_["value"].startswith("file:"):
                        tokens += self.tokenize_image(
                            PIL.Image.open(input_["value"].split(":", 1)[1])
                        )
                    else:
                        raise ValueError("Unknown image format.")
                elif isinstance(input_["value"], Image):
                    tokens += self.tokenize_image(input_["value"])
                else:
                    raise ValueError("Unknown image type.")
            elif input_["type"] == "sentinel":
                tokens += [
                    {
                        "<START-OF-IMAGE>": self.vocab.begin_image,
                        "<END-OF-TURN>": self.vocab.eot_id,
                    }[input_["value"]]
                ]
            elif input_["type"] == "ids":
                tokens += input_["value"]
            else:
                raise ValueError("Unknown input type.")
        return tokens

    def decode_text(self, ids: torch.LongTensor | list[list[int]]) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        for row, values in enumerate(ids):
            try:
                ids[row] = values[: values.index(self.vocab.eos_id)]
            except ValueError:
                pass

        return self.tokenizer.decode_batch(ids)

    def decode_image(self, ids: torch.LongTensor) -> list[PIL.Image]:
        return [self.pil_from_bpe_tokens(sample) for sample in ids]