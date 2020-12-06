from collections.abc import Iterable
import math

import torch
import torch.nn as nn

from .base import QNetwork
from ...utils.data import State
from .utils import xavier_initialization, orthogonal_initialization


class PortEmbed(nn.Module):
    """
    Create embedding from portfolio
    """
    def __init__(self, n_assets, d_model):
        super().__init__()
        self.embed = nn.Linear(n_assets, d_model)

    def forward(self, raw_port):
        return self.embed(raw_port)


class NormalHead(nn.Module):
    """
    For use in DQN style discrete agents
    """
    def __init__(self, d_model, output_shape):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.out = nn.Linear(d_model, self.n_assets * self.action_atoms)

    def forward(self, state_emb):
        qvals = self.out(state_emb).view(state_emb.shape[0], self.n_assets,
                                         self.action_atoms)
        return qvals


class DuelingHead(nn.Module):
    """
    For use in DQN style discrete agents
    """
    def __init__(self, d_model, output_shape):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.value_net = nn.Linear(d_model, self.n_assets)
        self.adv_net = nn.Linear(d_model, self.n_assets * self.action_atoms)

    def forward(self, state_emb):
        value = self.value_net(state_emb)
        adv = self.adv_net(state_emb).view(state_emb.shape[0], self.n_assets,
                                           self.action_atoms)
        qvals = value[..., None] + adv - adv.mean(-1, keepdim=True)
        return qvals


class Attention(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 attn_dropout,
                 pre_res_dropout,
                 max_seqlen,
                 head_dim=None):
        super().__init__()
        assert head_dim or d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(
                (max_seqlen, max_seqlen),
                dtype=torch.uint8)).view(1, 1, max_seqlen, max_seqlen))
        self.qkv_net = nn.Linear(d_model, 3 * n_heads * head_dim, bias=True)
        self.split_size = self.head_dim * self.n_heads
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_net = nn.Linear(n_heads * head_dim, bias=True)
        self.pre_res_drop = nn.Dropout(pre_res_dropout)

    def forward(self,
                x,
                memory=None,
                attn_mask=None,
                head_mask=None,
                output_attn=False):
        bs, seqlen = x.size()[:2]
        if memory is not None:
            x = torch.cat([memory, x], 1)
            x = self.qkv_net(x)
            q, k, v = x.split(self.split_size, dim=2)
            q = q[-seqlen:]
        else:
            x = self.qkv_net(x)
            q, k, v = x.split(self.split_size, dim=2)
        kvlen = v.size(1)
        q = q.view(bs, seqlen, self.n_heads,
                   self.head_dim).permute(0, 2, 1,
                                          3)  # batch, head, seq, head_dim
        k = k.view(bs, kvlen, self.n_heads,
                   self.head_dim).permute(0, 2, 3,
                                          1)  # batch, head, head_dim, seq
        v = v.view(bs, kvlen, self.n_heads,
                   self.head_dim).permute(0, 2, 1,
                                          3)  # batch, head, seq, head_dim
        a = torch.einsum(
            "bhqd,bhdk->bhqk",
            (q, k))  # product of queries with all keys batch, head, seq, seq
        a = a / math.sqrt(
            v.size(-1))  # normalizes using length of values/queries
        mask = self.mask[:, :, kvlen - seqlen:kvlen, :kvlen]  #
        a = torch.where(mask, a, -1e-4)  # masks out previous values
        if attn_mask is not None:  # extra attn mask if provided
            a += attn_mask
        a = self.attn_drop(nn.Softmax(dim=-1)(a))  # along past dimension
        if head_mask is not None:  # mask out heads if mask provided
            a = a * head_mask
        out = torch.einsum(
            'bhqk,bhkd->bhqd',
            (a,
             v))  # attn weights applied to values batch, head, seq, head_dim
        out = out.permute(0, 2, 1, 3).contiguous().view(
            bs, seqlen,
            self.n_heads * self.head_dim)  # batch, seq, head, head_dim
        out = self.pre_res_drop(self.out_net(out))
        outputs = [
            out,
        ]
        if output_attn:
            outputs += [a]
        return outputs


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.actv = nn.GELU()

    def forward(self, x):
        x = self.drop1(self.actv(self.fc1(x)))
        x = self.fc2(x)
        return self.drop2(x)


class Layer(nn.Module):
    """
    Layer unit of transformer
    In order, performs these computation:
    attention -> residual sum -> layer norm -
    -> fc projection -> residual sum -> layer norm
    """
    def __init__(self,
                 d_model,
                 d_ff,
                 n_heads,
                 attn_drop,
                 pre_res_drop,
                 mlp_drop,
                 max_seqlen=50,
                 norm_eps=1e-05):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.attn_drop = attn_drop
        self.pre_res_drop = pre_res_drop
        self.mlp_drop = mlp_drop
        self.max_seqlen = max_seqlen
        self.attn_block = Attention(self.d_model, self.n_heads, self.attn_drop,
                                    self.pre_res_drop, self.max_seqlen)
        self.mlp_block = MLP(self.d_model, self.d_ff, self.mlp_drop)
        self.norm1 = nn.LayerNorm(self.d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(self.d_model, eps=norm_eps)

    def forward(self, x, memory=None, attn_mask=None, head_mask=None):
        attn_outs = self.attn_block(x,
                                    memory=memory,
                                    attn_mask=attn_mask,
                                    head_mask=head_mask)
        a = attn_outs[0]
        x = self.norm1(x + a)
        mlp_out = self.mlp_block(x)
        x = self.norm2(x + mlp_out)
        outs = (x, ) + attn_outs[1:]
        return outs
