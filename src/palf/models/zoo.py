import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(self, input_len=48, horizon=12, input_dim=1, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation, padding=self.pad)
        self.conv2 = nn.Conv1d(ch, ch, kernel, dilation=dilation, padding=self.pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(self.act(self.conv1(x)[..., :-self.pad]))  # causal
        y = self.drop(self.act(self.conv2(y)[..., :-self.pad]))
        return x + y


class TCNForecaster(nn.Module):
    def __init__(self, input_len=48, horizon=12, input_dim=1, channels=64,
                 kernel=3, levels=4, dropout=0.0):
        super().__init__()
        self.inp = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.Sequential(*[TCNBlock(channels, kernel, 2 ** i, dropout)
                                      for i in range(levels)])
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(h[:, :, -1])


class TransformerForecaster(nn.Module):
    def __init__(self, input_len=48, horizon=12, input_dim=1, d_model=64,
                 nhead=4, layers=2, ff=128, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, input_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        h = self.encoder(self.inp(x) + self.pos)
        return self.head(h[:, -1])


class PatchTSTForecaster(nn.Module):
    """PatchTST channel-independent: moi bien (channel) duoc chia patch va dua qua
    CUNG MOT encoder dung chung trong so, xu ly doc lap tung kenh, roi ghep lai
    o lop dau ra de du bao bien muc tieu. Ho tro input_dim = d bat ky (d >= 1)."""

    def __init__(self, input_len=48, horizon=12, input_dim=1, patch_len=8, stride=4,
                 d_model=64, nhead=4, layers=2, ff=128, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (input_len - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(input_dim * self.n_patches * d_model, horizon)

    def forward(self, x):
        # x: (B, L, D) voi D = input_dim (D=1 la truong hop univariate cu)
        B, L, D = x.shape
        p = x.transpose(1, 2).unfold(2, self.patch_len, self.stride)  # (B, D, n, patch_len)
        p = p.reshape(B * D, self.n_patches, self.patch_len)  # gop batch+channel -> chia se trong so
        h = self.encoder(self.embed(p) + self.pos)            # (B*D, n, d_model)
        h = h.reshape(B, D * self.n_patches * h.shape[-1])     # ghep lai cac kenh o lop dau ra
        return self.head(h)


MODEL_REGISTRY = {
    "GRU": GRUForecaster,
    "TCN": TCNForecaster,
    "Transformer": TransformerForecaster,
    "PatchTST": PatchTSTForecaster,
}

MODEL_INFO = {
    "GRU": "RNN 2 lớp, nhẹ, làm baseline.",
    "TCN": "Conv nhân quả giãn nở (dilated), nhanh và ổn định.",
    "Transformer": "Encoder self-attention trên từng bước thời gian.",
    "PatchTST": "Chia chuỗi thành patch rồi đưa vào Transformer, mạnh cho chuỗi dài. Hỗ trợ đa biến (channel-independent).",
}


def build_model(name, input_len=48, horizon=12, input_dim=1, **kwargs):
    return MODEL_REGISTRY[name](input_len=input_len, horizon=horizon,
                                input_dim=input_dim, **kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)