import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from tqdm import trange, tqdm  # ✅ for progress bars
from rl.envs.simple_intersection import SimpleIntersection
from rl.agents.dqn import DQN

print("🚀 Starting RL training...")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'dqn.pt')

# -------------------------------------------------------------------------
# Replay Buffer
# -------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def add(self, *args):
        self.buf.append(tuple(args))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)

# -------------------------------------------------------------------------
# Training Function
# -------------------------------------------------------------------------
def train(episodes=300, batch_size=64, device='cpu'):
    env = SimpleIntersection(max_steps=200)
    obs_dim = 4
    act_dim = 2

    qnet = DQN(obs_dim, act_dim).to(device)
    target = DQN(obs_dim, act_dim).to(device)
    target.load_state_dict(qnet.state_dict())

    opt = optim.Adam(qnet.parameters(), lr=1e-3)
    buf = ReplayBuffer()
    gamma = 0.99
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    # ✅ tqdm progress bar for episodes
    for ep in trange(episodes, desc="Training Episodes", unit="episode"):
        s = env.reset()
        total = 0.0
        done = False

        # ✅ tqdm progress for steps within the episode (optional)
        with tqdm(total=env.max_steps, leave=False, desc=f"Ep {ep+1}") as pbar:
            while not done:
                if random.random() < eps:
                    a = random.randrange(act_dim)
                else:
                    with torch.no_grad():
                        t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                        a = int(qnet(t).argmax(1).item())

                s2, r, done, _ = env.step(a)
                buf.add(s, a, r, s2, done)
                s = s2
                total += r
                pbar.update(1)

                # Training step
                if len(buf) > batch_size:
                    ss, aa, rr, ss2, dd = buf.sample(batch_size)
                    ss = torch.tensor(ss, dtype=torch.float32).to(device)
                    aa = torch.tensor(aa, dtype=torch.int64).to(device)
                    rr = torch.tensor(rr, dtype=torch.float32).to(device)
                    ss2 = torch.tensor(ss2, dtype=torch.float32).to(device)
                    dd = torch.tensor(dd, dtype=torch.float32).to(device)

                    qvals = qnet(ss).gather(1, aa.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        qnext = target(ss2).max(1)[0]
                        y = rr + gamma * (1.0 - dd) * qnext

                    loss = (qvals - y).pow(2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # soft update target network
                    for p, tp in zip(qnet.parameters(), target.parameters()):
                        tp.data.mul_(0.995).add_(0.005 * p.data)

        eps = max(eps_min, eps * eps_decay)
        tqdm.write(f"Episode {ep+1}/{episodes} | Reward={total:.2f} | Eps={eps:.3f}")

        # Save model periodically
        if (ep + 1) % 10 == 0:
            torch.save(qnet.state_dict(), MODEL_PATH)
            tqdm.write(f"💾 Saved model to {MODEL_PATH}")

    # Final save
    torch.save(qnet.state_dict(), MODEL_PATH)
    tqdm.write(f"✅ Training complete. Final model saved to {MODEL_PATH}")

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == '__main__':
    train()

print("✅ Training completed successfully.")
