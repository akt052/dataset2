import pickle
import torch
from torch.utils.data import Dataset

class BabyAIDataset(Dataset):
    def __init__(self, path, vocab):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

        self.vocab = vocab

        for sample in self.data:
            vocab.add_sentence(sample["instruction"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        obs_seq = torch.tensor(sample["obs_seq"], dtype=torch.long)  # 🔥 FIXED
        act_seq = torch.tensor(sample["act_seq"], dtype=torch.long)

        instr = torch.tensor(
            self.vocab.encode(sample["instruction"]),
            dtype=torch.long
        )

        return {
            "obs_seq": obs_seq,
            "act_seq": act_seq,
            "instr": instr,
            "length": len(act_seq)
        }


def collate_fn(batch):
    max_T = max(x["length"] for x in batch)
    max_L = max(len(x["instr"]) for x in batch)

    obs_batch, act_batch, instr_batch, mask = [], [], [], []

    for item in batch:
        T = item["length"]
        L = len(item["instr"])

        obs = torch.zeros(max_T, 7, 7, 3, dtype=torch.long)
        act = torch.zeros(max_T, dtype=torch.long)
        instr = torch.zeros(max_L, dtype=torch.long)
        m = torch.zeros(max_T)

        obs[:T] = item["obs_seq"]
        act[:T] = item["act_seq"]
        instr[:L] = item["instr"]
        m[:T] = 1

        obs_batch.append(obs)
        act_batch.append(act)
        instr_batch.append(instr)
        mask.append(m)

    return {
        "obs_seq": torch.stack(obs_batch),
        "act_seq": torch.stack(act_batch),
        "instr": torch.stack(instr_batch),
        "mask": torch.stack(mask)
    }