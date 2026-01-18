import torch
from torch import nn

# Linear → ReLU → Linear → Dropout
# x -> Linear(512→2048) -> ReLU -> Linear(2048→512) -> Dropout

class feedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(0.1)

    
    def forward(self, x): # x shape: (batch_size, seq_len, d_model)
        out = self.fc1(x)
        out1 = self.relu(out)
        out2 = self.fc2(out1)
        out3 = self.dropout(out2)    

        return out3
    

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
    
        # Note: d_ff is usually 2/3 * original FFN dim in LLaMA
        # LLaMA does:
        # hidden_dim = int(2/3 * 4 * d_model)
        # d_model = 512
        # ReLU FFN = 2048
        # SwiGLU FFN ≈ 1365
        hidden_dim = int(2 / 3 * 4 * d_model)
        hidden_dim = (hidden_dim + 255) // 256 * 256

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x):
        gate = self.act(self.w2(x))
        out = self.w1(x) * gate
        out = self.w3(out)

        return out
    

# x = torch.randn(2, 128, 512)
# ffn = SwiGLU_FFN(512, 1365)
# y = ffn(x)
# print(y.shape)  # (2, 128, 512)



# testing 

# ffn = feedforward(d_model=512, d_ff=2048)
# x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)
# out = ffn(x)
# print(out.shape)  # → torch.Size([2, 10, 512])