import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class FiLMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.film = FiLM(out_ch, cond_dim)

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.bn(x)
        x = self.film(x, cond)
        return F.relu(x)


class BabyAIModel(nn.Module):
    def __init__(self, vocab_size, action_dim):
        super().__init__()

        # 🔥 SYMBOLIC EMBEDDINGS (EXACT FIX)
        self.obj_emb = nn.Embedding(20, 8)
        self.color_emb = nn.Embedding(10, 8)
        self.state_emb = nn.Embedding(5, 8)

        # instruction encoder
        self.word_emb = nn.Embedding(vocab_size, 64)
        self.gru = nn.GRU(64, 128, batch_first=True)

        cond_dim = 128

        self.block1 = FiLMBlock(24, 32, cond_dim)
        self.block2 = FiLMBlock(32, 64, cond_dim)

        self.lstm = nn.LSTM(64 * 7 * 7, 128, batch_first=True)
        self.fc = nn.Linear(128, action_dim)

    def forward(self, obs, instr, hidden=None):
        # ---- instruction ----
        emb = self.word_emb(instr)
        _, h = self.gru(emb)
        cond = h.squeeze(0)

        # ---- symbolic obs encoding ----
        obj = self.obj_emb(obs[..., 0])
        color = self.color_emb(obs[..., 1])
        state = self.state_emb(obs[..., 2])

        x = torch.cat([obj, color, state], dim=-1)  # (B,7,7,24)
        x = x.permute(0, 3, 1, 2)

        x = self.block1(x, cond)
        x = self.block2(x, cond)

        x = x.reshape(x.size(0), -1).unsqueeze(1)

        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1)

        logits = self.fc(x)

        return logits, hidden