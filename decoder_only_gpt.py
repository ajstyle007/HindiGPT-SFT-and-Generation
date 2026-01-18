import torch 
from torch import nn
from feed_forward_nn import feedforward, SwiGLU_FFN
from masked_mha import Masked_MHA
from rms_norm import RMSNorm
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# d_model = 512  # main model dimension
# num_heads = 8  # number of heads
# d_ff = 2048    # feedforward hidden dimension
# seq_len = 128  # max input length
# vocab_size = 30000



def generate_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    mask = (~mask).unsqueeze(0).unsqueeze(1)   # (1,1,L,L)
    return mask


class Decoder_GPT_Block(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, seq_len, dropout=0.1):
        super().__init__()

        # self.ffn = feedforward(d_model, d_ff)
        self.swi_glu = SwiGLU_FFN(d_model, d_ff)
        self.masked_mha = Masked_MHA(d_model, num_heads, max_seq_len=seq_len)

        self.rms_norm0 = RMSNorm(d_model)
        self.rms_norm1 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):

        h = self.rms_norm0(x)
        h = self.masked_mha(h, attention_mask=attention_mask)

        x = x + self.dropout(h)

        h = self.rms_norm1(x)

        h = self.swi_glu(h)

        x = x + self.dropout(h)

        return x

    # def forward(self, x, attention_mask=None):

    #     # ---- Attention block ----
    #     h = self.rms_norm0(x)

    #     h = checkpoint(
    #         self.masked_mha,
    #         h,
    #         attention_mask
    #     )

    #     x = x + self.dropout(h)

    #     # ---- FFN block ----
    #     h = self.rms_norm1(x)

    #     h = checkpoint(
    #         self.swi_glu,
    #         h
    #     )

    #     x = x + self.dropout(h)

    #     return x

    

class Decoder(nn.Module):
    def __init__(self,vocab_size, num_layers, d_model, d_ff, num_heads,seq_len, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [Decoder_GPT_Block(d_model, d_ff, num_heads, dropout) 
             for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.seq_len = seq_len

        self.register_buffer(
            "causal_mask",
            generate_subsequent_mask(seq_len)
        )
        # Original "Attention Is All You Need" paper did this
        # Har block ke baad tum already Add & Norm karte ho, lekin last block ke output me fir bhi thoda drift (distribution shift) aa jata hai.
        # Final LayerNorm output ko stabilize karta hai so that:
        # output distribution consistent ho
        # next layers (LM Head ya classifier) easily train ho
        # gradients stable rahe
    
    def forward_tokens(self, token_ids):
        return self.embedding(token_ids)

    def forward(self, x, attention_mask=None):
        """
        x       : (B, S_dec, D)
        enc_out : (B, S_enc, D)
        tgt_mask: causal mask (1,1,S_dec,S_dec)
        """

        B, S, D = x.shape

        attention_mask = self.causal_mask[:, :, :S, :S].to(x.device)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        return self.norm(x)

    

class My_GPT_model_SFT(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, d_ff, num_heads, seq_len, dropout=0.1):
        super().__init__()

        self.decoder = Decoder(
            vocab_size=vocab_size, num_layers=num_layers, d_model=d_model,
            d_ff=d_ff, num_heads=num_heads, seq_len=seq_len, dropout=dropout
        )

        # LM Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying 
        self.lm_head.weight = self.decoder.embedding.weight

    def forward(self, input_ids, attention_mask=None):
        """
        token_ids: (B, S)
        """

        # Token → Embedding
        x = self.decoder.forward_tokens(input_ids)  # (B, S, D)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]   # shape: (B, 1, 1, S)
            attn_mask = attn_mask.to(dtype=x.dtype)         # match dtype
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min   # 0 -> -inf, 1 -> 0


        # Decoder stack
        x = self.decoder(x, attention_mask=attn_mask)                          # (B, S, D)

        # LM Head → vocab logits
        logits = self.lm_head(x)                     # (B, S, V)

        return logits


class My_GPT_model(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, d_ff, num_heads, seq_len, dropout=0.1):
        super().__init__()

        self.decoder = Decoder(
            vocab_size=vocab_size, num_layers=num_layers, d_model=d_model,
            d_ff=d_ff, num_heads=num_heads, seq_len=seq_len, dropout=dropout
        )

        # LM Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying 
        self.lm_head.weight = self.decoder.embedding.weight

    def forward(self, token_ids):
        """
        token_ids: (B, S)
        """

        # Token → Embedding
        x = self.decoder.forward_tokens(token_ids)  # (B, S, D)

        # Decoder stack
        x = self.decoder(x)                          # (B, S, D)

        # LM Head → vocab logits
        logits = self.lm_head(x)                     # (B, S, V)

        return logits


# model = My_GPT_model(
#     vocab_size=30000,
#     num_layers=6,
#     d_model=512,
#     d_ff=2048,
#     num_heads=8,
#     seq_len=128
# )

# tokens = torch.randint(0, 30000, (2, 128))

# logits = model(tokens)

# print(logits.shape)
# # (2, 128, 30000)
# print(tokens)
# print("#################")
# print(logits)