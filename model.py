from dataclasses import dataclass

from typing import Tuple
import math
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelArgs:
    model_parallel_size: int = 1
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int | None = None
    vocab_size: int = -1
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    qk_normalization: bool = False
    swin_norm: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    hidden_dim: int = 1024

    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(
        self,
        model_parallel_size: int,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
    ):
        super().__init__()
        
        self.model_parallel_size = model_parallel_size
        self.head_dim = head_dim
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        
        self.wqkv = nn.Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
            dtype=torch.bfloat16, 
        )

        self.wo = nn.Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
            dtype=torch.bfloat16,
        )

        self.q_normalization = torch.nn.LayerNorm(head_dim)
        self.k_normalization = torch.nn.LayerNorm(head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape 

        xqkv = self.wqkv(x)
        xq = xqkv[:, :, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, :, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, 2)
        
        xq = xq.view(bsz, -1, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, -1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, -1, self.n_local_kv_heads, self.head_dim)
        
        # QK Normalization
        xq = self.q_normalization(xq)
        xk = self.k_normalization(xk)
        
        # ROPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xk = repeat_kv(xk, self.n_rep) 
        xv = repeat_kv(xv, self.n_rep) 

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3) / math.sqrt(self.head_dim))
        
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        
        if self.model_parallel_size > 1:
            dist.all_reduce(output, groupt=group)
        return output

class FeedForward(nn.Module):
    def __init__(
        self,
        model_parallel_size: int,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_muiltiplier: float | None = None,
    ):
        super().__init__()

        self.model_parallel_size = model_parallel_size
        
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_muiltiplier is not None:
            hidden_dim = int(ffn_dim_muiltiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % model_parallel_size == 0

        self.w13 = nn.Linear(
            dim,
            2 * hidden_dim // model_parallel_size,
            bias=False,
        )

        self.w2 = nn.Linear(
            hidden_dim //model_parallel_size,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        output = self.w2(F.silu(x1) * x3)
        if self.model_parallel_size > 1:
            dist.all_reduce(output, group=group)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        model_parallel_size = args.model_parallel_size
        assert args.n_heads % n_kv_heads == 0
        assert args.n_heads % model_parallel_size == 0
        assert n_kv_heads % model_parallel_size == 0
        
        self.attention = Attention(
            model_parallel_size=model_parallel_size,
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
        )
        self.feed_forward = FeedForward(
            model_parallel_size=model_parallel_size,
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_muiltiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.swin_norm = args.swin_norm
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        if self.swin_norm:
            h = x + self.attention_norm(
                self.attention.forward(
                    x,
                    freqs_cis=freqs_cis,
                    mask=mask,
                    group=group,
                )
            )
            out = h + self.ffn_norm(self.feed_forward(h, group=group))
        else:
            h = x + self.attention.forward(
                self.attention_norm(x),
                freqs_cis=freqs_cis,
                mask=mask,
                group=group,
            )
            out = h + self.feed_forward(self.ffn_norm(h), group=group)
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        torch.set_default_dtype(torch.bfloat16)
        self.args = args
        
        self.model_parallel_size = args.model_parallel_size
        assert args.dim % self.model_parallel_size == 0
        assert args.vocab_size > 0
        assert args.vocab_size % self.model_parallel_size == 0
        
        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim // self.model_parallel_size,
            dtype=torch.bfloat16,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        self.output = nn.Linear(
            args.dim,
            args.vocab_size // self.model_parallel_size,
            bias=False,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            self.args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        group: dist.ProcessGroup | None = None,
    ):
        bsz, seqlen = tokens.shape
        device = tokens.device
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        """
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=device
            )
            mask = mask.to(torch.float32).triu(diagonal=start_pos+1).type_as(h)
        """

        if self.model_parallel_size > 1:
            gather = [torch.empty_like(h) for _ in range(self.model_parallel_size)]
            dist.all_gather(gather, h, group=group)
            h = torch.cat(gather, dim=-1)
        
        layer_output = []
            
        for i, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, (mask.to(device) if mask is not None else mask))
            layer_output.append(h[:,0,:])

        """
        logits = self.output(self.norm(h))
        if self.model_parallel_size > 1:
            gather = [torch.empty_like(logits) for _ in range(self.model_parallel_size)]
            dist.all_gather(gather, logits, group=group)
            logits = torch.cat(gather, dim=-1)
        return logits.float()
        """
        return torch.cat(layer_output, dim=-1)
            

class GRUCell(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.input_dim = args.dim
        self.hidden_dim = args.hidden_dim 
        self.x2h = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.h2h = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.s2h = nn.Linear(args.dim * args.n_layers, self.hidden_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden, summary):
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3,1)
        h_r, h_i, h_n = gate_h.chunk(3,1)
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n) + self.s2h(summary))

        hy = (1-inputgate) * hidden + inputgate * newgate
        return hy

class GRU(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.hidden_dim = args.hidden_dim
        self.gru_cell = GRUCell(args)
        self.fc = nn.Linear(args.hidden_dim, args.vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
            dtype=torch.bfloat16,
        )

        self.hidden = None

    def train_forward(self, x, summary, h0 = None):
        if h0 is None:
            h0 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        
        outs = []
        hn = h0[:,:]
        x = self.tok_embeddings(x)
        
        for seq in range(x.size(1)):
            hn = self.gru_cell.forward(x[:, seq, :], hn, summary) 
            outs.append(hn.unsqueeze(1))
        out = torch.cat(outs, dim=1)
        out = self.fc(out)
        out = self.softmax(out)
        return out
    
    def generate(self, x, summary, sos=False):
        if sos:
            self.hidden = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        assert self.hidden is not None

        x = self.tok_embeddings(x)
        self.hidden = self.gru_cell(x, self.hidden, summary)
        out = self.fc(self.hidden)
        out = self.softmax(out)
        return out

    def reset(self):
        self.hidden = None
        

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.transformer = Transformer(args)
        self.gru = GRU(args)
    
    def train_forward(
        self,
        input: torch.Tensor,
        y: torch.Tensor,
        group: dist.ProcessGroup | None = None 
    ):
        summary = self.transformer(input, 0, group)
        output = self.gru.train_forward(y, summary)
        return output

    def generate(
        self,
        input: torch.Tensor,
        prev: torch.Tensor,
        sos: bool = False,
        group: dist.ProcessGroup | None = None
    ):
        summary = self.transformer(input, 0, group)
        output = self.gru.generate(prev, summary, sos)
        return output
