import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ── System parameters ─────────────────────────────────────────────────────────
M     = 1.0
kv    = 0.5
dt    = 0.1
F_max = 2.0

def step(x, v, F):
    F = np.clip(F, -F_max, F_max)
    a = (F - kv*v) / M
    v = v + a*dt
    x = x + v*dt
    return x, v

# ── Networks ──────────────────────────────────────────────────────────────────
actor = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 1)
)
critic = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 1)
)

actor_opt   = optim.Adam(actor.parameters(),  lr=0.0003)
critic_opt  = optim.Adam(critic.parameters(), lr=0.001)

# Fixed small std — more stable than learnable for this problem
ACTION_STD = 0.3

def get_action_stochastic(state_t):
    mean   = actor(state_t)
    dist   = torch.distributions.Normal(mean, ACTION_STD)
    raw    = dist.sample()
    action = F_max * torch.tanh(raw)
    lp     = dist.log_prob(raw) - torch.log(1 - torch.tanh(raw)**2 + 1e-6)
    return action, lp

def get_action_det(x, v):
    state_t = torch.tensor([x/8.0, v/3.0], dtype=torch.float32)
    with torch.no_grad():
        return (F_max * torch.tanh(actor(state_t))).item()

# ── Training ──────────────────────────────────────────────────────────────────
gamma           = 0.99
episode_rewards = []

for episode in range(5000):

    scale = min(1.0, episode / 2000)
    x = np.random.uniform(-2 - 6*scale, 2 + 6*scale)
    v = np.random.uniform(-(1 + 1*scale), 1 + 1*scale)

    states, log_probs, rewards = [], [], []

    for t in range(150):
        state_np = np.array([x/8.0, v/3.0], dtype=np.float32)
        state_t  = torch.tensor(state_np)

        action, lp = get_action_stochastic(state_t)
        F = action.item()
        x, v = step(x, v, F)

        reward = -(x**2 + 0.1*v**2 + 0.01*F**2)
        reward = max(reward, -50.0)

        states.append(state_np)
        log_probs.append(lp)
        rewards.append(reward)

    # ── Returns ───────────────────────────────────────────────────────────────
    R, returns = 0, []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns_t = torch.tensor(returns, dtype=torch.float32)
    states_t  = torch.tensor(np.array(states))

    # ── Critic update ─────────────────────────────────────────────────────────
    values      = critic(states_t).squeeze()
    critic_loss = nn.MSELoss()(values, returns_t)
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
    critic_opt.step()

    # ── Advantages ────────────────────────────────────────────────────────────
    with torch.no_grad():
        values_new = critic(states_t).squeeze()
    advantages = returns_t - values_new
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── Actor update ──────────────────────────────────────────────────────────
    lp_t       = torch.stack(log_probs).squeeze()
    actor_loss = -(lp_t * advantages).sum()
    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    actor_opt.step()

    total_r = sum(rewards)
    episode_rewards.append(total_r)

    if episode % 500 == 0:
        print(f"episode {episode:4d}  final x={x:7.3f}  reward={total_r:8.1f}")

# ── Save weights ──────────────────────────────────────────────────────────────
for layer_idx, name in [(0,'1'), (2,'2'), (4,'3')]:
    np.savetxt(f"W{name}.txt", actor[layer_idx].weight.detach().numpy())
    np.savetxt(f"b{name}.txt", actor[layer_idx].bias.detach().numpy().reshape(1,-1))
print("Weights saved: W1,W2,W3, b1,b2,b3")

# ── Simulation ────────────────────────────────────────────────────────────────
x, v = 7.0, 0.0
xs, vs, Fs, ts = [], [], [], []
for t in range(300):
    F = get_action_det(x, v)
    x, v = step(x, v, F)
    xs.append(x); vs.append(v); Fs.append(F); ts.append(t*dt)

# ── Plots — fixed axhline syntax ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Closed-loop: Actor-Critic controller")

axes[0].plot(ts, xs, 'b')
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.8)   # fixed
axes[0].set_ylabel('Position x'); axes[0].grid(True)

axes[1].plot(ts, vs, 'g')
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.8)   # fixed
axes[1].set_ylabel('Velocity v'); axes[1].grid(True)

axes[2].plot(ts, Fs, 'r')
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8)   # fixed
axes[2].set_ylabel(f'Force F (±{F_max})')
axes[2].set_xlabel('Time (s)'); axes[2].grid(True)

plt.tight_layout()

# Training progress
plt.figure()
w = 100
plt.plot(episode_rewards, alpha=0.2, color='blue', label='raw')
if len(episode_rewards) >= w:
    smoothed = np.convolve(episode_rewards, np.ones(w)/w, mode='valid')
    plt.plot(range(w-1, len(episode_rewards)), smoothed, 'r', lw=2, label='smoothed')
plt.xlabel('Episode'); plt.ylabel('Total reward')
plt.title('Training progress'); plt.legend(); plt.grid(True)
plt.show()

print("\n── MATLAB forward pass ──")
print("state_norm = [x/8.0; v/3.0];")
print("h1     = tanh(W1 * state_norm + b1');")
print("h2     = tanh(W2 * h1         + b2');")
print("mean_v =      W3 * h2         + b3';")
print("action = F_max * tanh(mean_v);")