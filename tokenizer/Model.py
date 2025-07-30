import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import pickle
from functools import cached_property
from tokenizer.TextGlow import TextGlow
import scipy.linalg as la
from math import log, pi

class Tlow(nn.Module):
    def __init__(self, llm_embedding_dim, n_flow, n_block) -> None:
        self._config = locals()
        super().__init__()
        self.glow = TextGlow(
            in_dim=llm_embedding_dim, 
            n_flow=n_flow, 
            n_block=n_block, 
            affine=True
        )

    @cached_property
    def config(self) -> dict:
        return self._config
    
    @torch.compile(options={"triton.cudagraphs": False})
    def forward(self, batch):
        x = batch.x
        log_p_sum, logdet, z_out = self.glow(x)
        log_p_sum = log_p_sum.mean()
        logdet = logdet.mean()
        nll = -(log_p_sum + logdet)
        return TlowComputedLosses(nll, nll, log_p_sum, logdet), torch.cat(z_out, dim=-1)

    @property
    def device(self) -> torch.device:
        return next(self.glow.parameters()).device


from typing import NamedTuple
from torch import Tensor
class TlowComputedLosses(NamedTuple):
    loss: Tensor
    nll: Tensor
    log_p: Tensor
    log_det: Tensor