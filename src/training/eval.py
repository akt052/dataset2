import torch
import gymnasium as gym
import minigrid

from minigrid.utils.baby_ai_bot import BabyAIBot
from src.models.babyai_model import BabyAIModel
from src.utils.vocab import Vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_with_agreement(model, env, vocab, episodes=5):
    model.eval()
    average = 0

    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            bot = BabyAIBot(env)

            instr = torch.tensor(
                [vocab.encode(env.mission)],
                dtype=torch.long
            ).to(DEVICE)

            hidden = None
            done = False

            total_steps = 0
            matches = 0

            while not done:
                obs_t = torch.tensor(
                    obs["image"],
                    dtype=torch.long
                ).unsqueeze(0).to(DEVICE)

                # model action
                logits, hidden = model(obs_t, instr, hidden)
                model_action = logits.argmax(dim=1).item()

                # bot action
                try:
                    bot_action = bot.replan(obs)
                except AssertionError:
                    break

                # compare
                if model_action == bot_action:
                    matches += 1

                total_steps += 1

                # step environment using MODEL action
                obs, reward, terminated, truncated, _ = env.step(model_action)
                done = terminated or truncated

            agreement = matches / total_steps if total_steps > 0 else 0

            print(f"Episode {ep+1}:")
            print(f"  Steps: {total_steps}")
            print(f"  Agreement: {agreement:.3f}")
            print("-" * 30)
            average+= agreement
    average /= episodes
    print(f"Average Agreement: {average:.3f}")

if __name__ == "__main__":
    # ---- vocab ----
    vocab = Vocab()
    import pickle
    data = pickle.load(open("data/demos/gotolocal_seq.pkl", "rb"))
    for sample in data:
        vocab.add_sentence(sample["instruction"])

    # ---- model ----
    model = BabyAIModel(len(vocab), 7).to(DEVICE)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))

    # ---- env ----
    env = gym.make("BabyAI-GoToLocal-v0").unwrapped

    # ---- run ----
    evaluate_with_agreement(model, env, vocab, episodes=1000)