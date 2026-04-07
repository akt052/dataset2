import torch
import gymnasium as gym
import minigrid 
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# TRAINING (Imitation Learning + TBPTT)
# --------------------------------------------------
def train_il(model, dataloader, epochs=20):
    model = model.to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-5
    )

    criterion = nn.CrossEntropyLoss(reduction="none")

    TBPTT_STEPS = 20  

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            obs_seq = batch["obs_seq"].to(DEVICE)    
            act_seq = batch["act_seq"].to(DEVICE)    
            instr = batch["instr"].to(DEVICE)        
            mask = batch["mask"].to(DEVICE)          

            B, T = act_seq.shape

            hidden = None

            for t in range(0, T, TBPTT_STEPS):
                obs_chunk = obs_seq[:, t:t+TBPTT_STEPS]
                act_chunk = act_seq[:, t:t+TBPTT_STEPS]
                mask_chunk = mask[:, t:t+TBPTT_STEPS]

                optimizer.zero_grad()
                loss = 0

                for step in range(obs_chunk.size(1)):
                    obs = obs_chunk[:, step]              # (B,7,7,3)
                    target = act_chunk[:, step]           # (B)
                    step_mask = mask_chunk[:, step]       # (B)

                    logits, hidden = model(obs, instr, hidden)

                    step_loss = criterion(logits, target)
                    step_loss = step_loss * step_mask

                    loss += step_loss.mean()

                loss = loss / obs_chunk.size(1)

                loss.backward()
                optimizer.step()

                # 🔥 TBPTT detach
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden)

                total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("✅ Model saved as model.pt")


# --------------------------------------------------
# EVALUATION (SUCCESS RATE)
# --------------------------------------------------
def evaluate(model, env, vocab, episodes=100):
    model.eval()
    success = 0

    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()

            instr = torch.tensor(
                [vocab.encode(env.mission)],
                dtype=torch.long
            ).to(DEVICE)

            hidden = None
            done = False

            while not done:
                obs_t = torch.tensor(
                    obs["image"],
                    dtype=torch.long
                ).unsqueeze(0).to(DEVICE)

                logits, hidden = model(obs_t, instr, hidden)
                action = logits.argmax(dim=1).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            if reward > 0:
                success += 1

    success_rate = success / episodes
    print(f"🎯 Success Rate: {success_rate:.3f}")

    return success_rate


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    import gymnasium as gym
    from torch.utils.data import DataLoader

    from src.training.dataset_loader import BabyAIDataset, collate_fn
    from src.utils.vocab import Vocab
    from src.models.babyai_model import BabyAIModel

    # ---- setup ----
    vocab = Vocab()

    dataset = BabyAIDataset(
        path="data/demos/gotolocal_seq.pkl",
        vocab=vocab
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = BabyAIModel(
        vocab_size=len(vocab),
        action_dim=7
    )

    # ---- train ----
    train_il(model, dataloader, epochs=20)

    # ---- evaluate ----
    env = gym.make("BabyAI-GoToLocal-v0").unwrapped
    evaluate(model, env, vocab, episodes=100)