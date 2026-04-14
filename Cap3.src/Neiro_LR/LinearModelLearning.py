import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -- Reproducibility ----------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -- Vessel parameters --------------------------------------------------------
M        = 702000.0
b        = 6970.0
F_max    = 150000.0
V_max    = 10.0
dt       = 0.1
tau      = M / b
T_ep     = 600
N_steps  = int(T_ep / dt)   # 6000 steps per episode
N_ROLL   = 100               # open-loop validation window = 10 seconds

print(f"Vessel time constant tau = {tau:.1f} s")
print(f"Random seed              = {SEED}")

# -- True linear vessel -------------------------------------------------------
def vessel_step(V, F_control):
    dVdt   = (F_control - b * V) / M
    V_next = V + dVdt * dt
    return V_next

# -- Control signal -----------------------------------------------------------
def generate_control(n_steps, dt, F_max, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(n_steps) * dt
    F = np.zeros(n_steps)
    for _ in range(np.random.randint(3, 6)):
        period    = np.random.uniform(10.0, 600.0)
        amplitude = np.random.uniform(0.1, 1.0) * F_max
        phase     = np.random.uniform(0, 2*np.pi)
        F        += amplitude * np.sin(2*np.pi/period * t + phase)
    return np.clip(F, -F_max, F_max)

# -- Open-loop rollout validation ---------------------------------------------
def openloop_rmse(model, ep_list, n_roll=N_ROLL):
    """
    Run model open-loop for n_roll steps from multiple starting points.
    Tests BOTH positive and negative velocity regions.
    Returns mean absolute error at end of rollout.
    """
    model.eval()
    errors = []
    for F_profile, V_profile in ep_list[:10]:
        # test from multiple points in episode -- both + and - V regions
        for t0 in range(0, N_steps - n_roll - 1, 500):
            V_nn = V_profile[t0]
            for k in range(n_roll):
                with torch.no_grad():
                    inp  = torch.tensor([V_nn / V_max,
                                         F_profile[t0+k] / F_max],
                                        dtype=torch.float32)
                    V_nn = model(inp).item() * V_max
            errors.append(abs(V_nn - V_profile[t0 + n_roll]))
    model.train()
    return np.mean(errors), np.max(errors)

# -- Network ------------------------------------------------------------------
def make_model():
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 32), nn.Tanh(),
        nn.Linear(32, 1)
    )

model = make_model()

# -- Generate training data ---------------------------------------------------
# Symmetric: equal number of episodes starting positive and negative V
# Also ensure F profile covers both + and - force regions symmetrically
print("Generating training data...")
N_episodes_data = 40   # doubled for better coverage
all_inputs, all_targets = [], []
episodes = []          # keep full episodes for open-loop validation

for ep in range(N_episodes_data):
    # Alternate positive/negative starts for symmetry
    if ep % 2 == 0:
        V = np.random.uniform(0.0, 2.0)
    else:
        V = np.random.uniform(-2.0, 0.0)

    F_profile = generate_control(N_steps, dt, F_max)

    # Mirror every other episode F profile to ensure - force coverage
    if ep % 4 >= 2:
        F_profile = -F_profile

    V_profile = np.zeros(N_steps)
    V_profile[0] = V
    for t in range(N_steps - 1):
        V_next = vessel_step(V, F_profile[t])
        all_inputs.append( [V       / V_max,  F_profile[t] / F_max])
        all_targets.append([V_next  / V_max])
        V = V_next
        V_profile[t+1] = V

    episodes.append((F_profile, V_profile))

X_train = torch.tensor(all_inputs,  dtype=torch.float32)
Y_train = torch.tensor(all_targets, dtype=torch.float32)
print(f"Training samples: {len(X_train):,}")
print(f"V range in training data: [{min(a[0] for a in all_inputs)*V_max:.2f}, "
      f"{max(a[0] for a in all_inputs)*V_max:.2f}] m/s")
print(f"F range in training data: [{min(a[1] for a in all_inputs)*F_max:.0f}, "
      f"{max(a[1] for a in all_inputs)*F_max:.0f}] N")

# -- Training -----------------------------------------------------------------
print("\nTraining...")
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

BATCH_SIZE           = 4096
EPOCHS               = 500
best_openloop_mean   = float('inf')
best_singlestep_loss = float('inf')
patience_ctr         = 0
EARLY_STOP           = 40
losses               = []
ol_means             = []
ol_maxs              = []

dataset = torch.utils.data.TensorDataset(X_train, Y_train)
loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       generator=torch.Generator().manual_seed(SEED))

for epoch in range(EPOCHS):
    epoch_loss = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = nn.MSELoss()(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    losses.append(avg_loss)
    scheduler.step(avg_loss)

    # Open-loop validation every 10 epochs
    if epoch % 10 == 0:
        ol_mean, ol_max = openloop_rmse(model, episodes)
        ol_means.append(ol_mean)
        ol_maxs.append(ol_max)

        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}  "
                  f"loss={avg_loss:.7f}  "
                  f"ss_RMSE={np.sqrt(avg_loss)*V_max:.4f} m/s  "
                  f"OL_mean={ol_mean:.4f} m/s  "
                  f"OL_max={ol_max:.4f} m/s")

        # Save based on open-loop performance -- this is what matters
        if ol_mean < best_openloop_mean:
            best_openloop_mean = ol_mean
            torch.save(model.state_dict(), 'best_model.pt')
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP:
                print(f"  Early stopping at epoch {epoch}")
                break

model.load_state_dict(torch.load('best_model.pt'))
ol_mean_final, ol_max_final = openloop_rmse(model, episodes)
print(f"\nBest model:  OL_mean={ol_mean_final:.4f} m/s  OL_max={ol_max_final:.4f} m/s")

# -- Save weights -------------------------------------------------------------
for idx, name in [(0,'1'),(2,'2'),(4,'3'),(6,'4')]:
    np.savetxt(f"W{name}.txt", model[idx].weight.detach().numpy())
    np.savetxt(f"b{name}.txt", model[idx].bias.detach().numpy().reshape(1,-1))
print("Weights saved: W1,W2,W3,W4  b1,b2,b3,b4")

# -- Detailed open-loop test --------------------------------------------------
print("\nDetailed 10s open-loop test (matches MATLAB diagnostic)...")
F_val = generate_control(N_steps, dt, F_max, seed=99)
for V_start in [0.0, 0.2, -0.2, 1.0, -1.0]:
    V_nn   = V_start
    V_true = V_start
    for k in range(N_ROLL):
        V_true = vessel_step(V_true, F_val[k])
        with torch.no_grad():
            inp  = torch.tensor([V_nn/V_max, F_val[k]/F_max],
                                 dtype=torch.float32)
            V_nn = model(inp).item() * V_max
    print(f"  V_start={V_start:+.1f}  V_true={V_true:.4f}  "
          f"V_nn={V_nn:.4f}  bias={V_nn-V_true:+.4f}")

# -- Plots --------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].semilogy(losses, 'b', lw=1.0, label='Single-step loss')
axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True)
axes[0].set_title('Training loss')

ax2 = axes[0].twinx()
ax2.plot(range(0, len(ol_means)*10, 10), ol_means, 'r', lw=1.5, label='OL mean error')
ax2.plot(range(0, len(ol_maxs)*10,  10), ol_maxs,  'm', lw=1.0, label='OL max error')
ax2.set_ylabel('Open-loop error (m/s)', color='r')
ax2.legend(loc='upper right')

# Symmetry check -- plot V_nn vs V_true for range of inputs
V_test_range = np.linspace(-3, 3, 200)
F_test       = 0.0
V_nn_out     = []
V_true_out   = []
for V_t in V_test_range:
    with torch.no_grad():
        inp = torch.tensor([V_t/V_max, F_test/F_max], dtype=torch.float32)
        V_nn_out.append(model(inp).item() * V_max)
    V_true_out.append(vessel_step(V_t, F_test))

axes[1].plot(V_test_range, V_true_out, 'b', lw=2.0, label='True vessel (F=0)')
axes[1].plot(V_test_range, V_nn_out,   'r--', lw=1.5, label='Neural model (F=0)')
axes[1].set_xlabel('V input (m/s)'); axes[1].set_ylabel('V_next (m/s)')
axes[1].set_title('Symmetry check: V_next vs V_input at F=0')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.show()

print("\n-- MATLAB forward pass --")
print(f"V_max = {V_max};  F_max = {F_max};")
print("inp    = [V / V_max;  F / F_max];")
print("h1     = tanh(W1 * inp + b1');")
print("h2     = tanh(W2 * h1  + b2');")
print("h3     = tanh(W3 * h2  + b3');")
print("V_next =      W4 * h3  + b4';")
print("V_next = V_next * V_max;")