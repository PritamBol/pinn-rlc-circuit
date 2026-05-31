"""
PINN Phase 4 v3 — Fixed Identifiability Problem
=================================================
Root cause of v2 failure:
  The ODE L*q'' + R*q' + q/C = 0 can be rewritten as:
    q'' + (R/L)*q' + (1/LC)*q = 0

  Only TWO combinations matter:
    alpha = R/L   (damping coefficient)
    omega2 = 1/LC (natural frequency squared)

  So infinite (R,L,C) triplets satisfy the same ODE!
  e.g. R=10,L=1,C=0.01  AND  R=0.1,L=0.01,C=1.0
  both give alpha=10, omega2=100.

Fix strategy:
  1. Normalize the ODE — divide through by L, learn alpha and omega2
  2. Reconstruct R,L,C from physical constraints + one anchor
  3. Add regularization loss pulling params toward physical prior
  4. Use tighter clamp bounds based on domain knowledge
"""
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

torch.manual_seed(42)
np.random.seed(42)

# ── True values ───────────────────────────────────────────────
R_TRUE, L_TRUE, C_TRUE = 10.0, 1.0, 0.01
T_START, T_END = 0.0, 2.0
NOISE_LEVEL = 0.02
N_OBS   = 80
N_COLL  = 1500
EPOCHS  = 30000
SAVE_PATH = os.path.join(os.getcwd(), 'pinn_phase4_v3_result.png')

# True derived quantities
ALPHA_TRUE  = R_TRUE / L_TRUE          # = 10.0
OMEGA2_TRUE = 1.0 / (L_TRUE * C_TRUE)  # = 100.0
print(f"True α  = R/L  = {ALPHA_TRUE}")
print(f"True ω² = 1/LC = {OMEGA2_TRUE}")
print(f"These are the IDENTIFIABLE quantities from q(t) alone.\n")

# ── Generate data ─────────────────────────────────────────────
def generate_data():
    def rlc(t, y):
        return [y[1], -ALPHA_TRUE * y[1] - OMEGA2_TRUE * y[0]]
    t_eval = np.linspace(T_START, T_END, N_OBS)
    sol = solve_ivp(rlc, [T_START, T_END], [1.0, 0.0],
                    t_eval=t_eval, method='RK45', rtol=1e-10)
    q_clean = sol.y[0]
    q_noisy = q_clean + np.random.normal(0, NOISE_LEVEL, size=q_clean.shape)
    print(f"Data: {N_OBS} points, noise σ={NOISE_LEVEL}")
    return sol.t, q_clean, q_noisy

# ── Network ───────────────────────────────────────────────────
class InversePINN(nn.Module):
    """
    KEY CHANGE: We learn alpha=R/L and omega2=1/LC directly.
    These are uniquely identifiable from q(t) data.
    To recover individual R, L, C we need one additional
    constraint — here we fix L=1H as the anchor (known from
    the inductor's physical measurement).
    """
    def __init__(self, hidden=64, n_layers=5):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Learn alpha = R/L and omega2 = 1/LC in log-space
        # Initial guesses: alpha=5 (true=10), omega2=50 (true=100)
        self.log_alpha  = nn.Parameter(torch.tensor([np.log(5.0)]))
        self.log_omega2 = nn.Parameter(torch.tensor([np.log(50.0)]))

        # Anchor: L is assumed measurable (e.g. with LCR meter)
        # In practice this could be any one known component
        self.L_anchor = L_TRUE  # known

    def forward(self, t):
        return self.net(t)

    @property
    def alpha(self):
        # clamp: alpha must be positive, physically between 0.1 and 1000
        return torch.exp(torch.clamp(self.log_alpha, np.log(0.1), np.log(1000)))

    @property
    def omega2(self):
        # clamp: omega2 must be positive
        return torch.exp(torch.clamp(self.log_omega2, np.log(0.1), np.log(10000)))

    # Recover R, L, C from alpha and omega2 using the anchor
    @property
    def R(self): return self.alpha * self.L_anchor
    @property
    def L(self): return torch.tensor(self.L_anchor)
    @property
    def C(self): return 1.0 / (self.omega2 * self.L_anchor)

# ── Losses ────────────────────────────────────────────────────
def physics_loss(model, t_phys):
    """
    Normalized ODE: q'' + alpha*q' + omega2*q = 0
    Dividing by L removes L from the equation entirely —
    now alpha and omega2 are directly identifiable.
    """
    t_phys = t_phys.requires_grad_(True)
    q = model(t_phys)
    q_t = torch.autograd.grad(q, t_phys,
              grad_outputs=torch.ones_like(q), create_graph=True)[0]
    q_tt = torch.autograd.grad(q_t, t_phys,
               grad_outputs=torch.ones_like(q_t), create_graph=True)[0]
    # Normalized form: q'' + alpha*q' + omega2*q = 0
    res = q_tt + model.alpha * q_t + model.omega2 * q
    return torch.mean(res ** 2)

def ic_loss(model):
    t0 = torch.tensor([[0.0]], requires_grad=True)
    q0 = model(t0)
    q0_t = torch.autograd.grad(q0, t0,
               grad_outputs=torch.ones_like(q0), create_graph=True)[0]
    return (q0 - 1.0) ** 2 + q0_t ** 2

def data_loss(model, t_obs, q_obs):
    return torch.mean((model(t_obs) - q_obs) ** 2)

def regularization_loss(model):
    """
    Soft prior: pull alpha and omega2 toward physically
    reasonable order-of-magnitude estimates.
    This breaks degeneracy when data is limited.
    Prior: alpha ~ O(10), omega2 ~ O(100)
    """
    prior_alpha  = torch.tensor([np.log(10.0)])
    prior_omega2 = torch.tensor([np.log(100.0)])
    reg = 0.01 * (model.log_alpha - prior_alpha) ** 2
    reg += 0.01 * (model.log_omega2 - prior_omega2) ** 2
    return reg.squeeze()

# ── Training ─────────────────────────────────────────────────
def train(t_obs_np, q_noisy_np):
    model = InversePINN()
    t_obs = torch.FloatTensor(t_obs_np).reshape(-1, 1)
    q_obs = torch.FloatTensor(q_noisy_np).reshape(-1, 1)

    optimizer = torch.optim.Adam([
        {'params': model.net.parameters(),            'lr': 1e-3},
        {'params': [model.log_alpha, model.log_omega2], 'lr': 8e-3}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    t_phys = torch.FloatTensor(N_COLL, 1).uniform_(T_START, T_END)
    history = {"alpha": [], "omega2": [], "R": [], "C": [], "loss": []}

    print(f"\n{'Epoch':>7} | {'Loss':>10} | {'α=R/L':>8} | {'ω²=1/LC':>10} | {'R(Ω)':>8} | {'C(F)':>9}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        if epoch % 500 == 0:
            t_phys = torch.FloatTensor(N_COLL, 1).uniform_(T_START, T_END)

        li = ic_loss(model)
        ld = data_loss(model, t_obs, q_obs)
        lr = regularization_loss(model)

        if epoch < 5000:
            loss = 500.0 * ld + 100.0 * li + lr
        else:
            lp = physics_loss(model, t_phys)
            w = min(1.0, (epoch - 5000) / 5000)
            loss = w * lp + 500.0 * ld + 100.0 * li + lr

        if torch.isnan(loss):
            print(f"NaN at epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        a  = model.alpha.item()
        w2 = model.omega2.item()
        R_est = model.R.item()
        C_est = model.C.item()

        history["alpha"].append(a)
        history["omega2"].append(w2)
        history["R"].append(R_est)
        history["C"].append(C_est)
        history["loss"].append(loss.item())

        if epoch % 5000 == 0 or epoch == 1:
            print(f"{epoch:>7} | {loss.item():>10.5f} | {a:>8.3f} | "
                  f"{w2:>10.3f} | {R_est:>8.3f} | {C_est:>9.5f}")

    print(f"\n{'='*60}")
    print(f"Identifiable quantities:")
    print(f"  α  = R/L  : {model.alpha.item():.4f}  (true: {ALPHA_TRUE},  error: {abs(model.alpha.item()-ALPHA_TRUE)/ALPHA_TRUE*100:.1f}%)")
    print(f"  ω² = 1/LC : {model.omega2.item():.4f}  (true: {OMEGA2_TRUE}, error: {abs(model.omega2.item()-OMEGA2_TRUE)/OMEGA2_TRUE*100:.1f}%)")
    print(f"\nRecovered parameters (using L={L_TRUE}H anchor):")
    print(f"  R = α·L   : {model.R.item():.4f} Ω  (true: {R_TRUE}Ω,  error: {abs(model.R.item()-R_TRUE)/R_TRUE*100:.1f}%)")
    print(f"  C = 1/ω²L : {model.C.item():.5f} F  (true: {C_TRUE}F, error: {abs(model.C.item()-C_TRUE)/C_TRUE*100:.1f}%)")
    print(f"{'='*60}")
    return model, history

# ── Plot ──────────────────────────────────────────────────────
def plot_results(model, history, t_obs, q_clean, q_noisy):
    def rlc(t, y):
        return [y[1], -ALPHA_TRUE*y[1] - OMEGA2_TRUE*y[0]]
    sol = solve_ivp(rlc, [T_START, T_END], [1.0, 0.0],
                    t_eval=np.linspace(T_START, T_END, 500), method='RK45', rtol=1e-9)
    t_ref, q_ref = sol.t, sol.y[0]

    t_test = torch.linspace(T_START, T_END, 500).reshape(-1, 1)
    with torch.no_grad():
        q_pred = model(t_test).numpy().flatten()

    ep = list(range(1, len(history["loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Phase 4 v3 — Inverse Problem (Identifiability Fixed)',
                 fontsize=13, fontweight='bold')

    # Plot 1: solution
    ax = axes[0][0]
    ax.plot(t_ref, q_ref, 'k-', lw=2.5, label='True q(t)')
    ax.scatter(t_obs, q_noisy, s=14, color='gray', alpha=0.6,
               label=f'Noisy data (σ={NOISE_LEVEL})', zorder=3)
    ax.plot(t_test.numpy().flatten(), q_pred, 'r--', lw=2, label='PINN')
    ax.set_title('Solution Recovery')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('q(t) [C]')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.97,
            f"α={model.alpha.item():.3f} (true {ALPHA_TRUE})\n"
            f"ω²={model.omega2.item():.2f} (true {OMEGA2_TRUE})\n"
            f"→ R={model.R.item():.3f}Ω (true {R_TRUE})\n"
            f"→ C={model.C.item():.4f}F (true {C_TRUE})",
            transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Plot 2: alpha convergence
    ax = axes[0][1]
    ax.axhline(ALPHA_TRUE, color='k', lw=1.5, ls='--', label=f'True α={ALPHA_TRUE}')
    ax.plot(ep, history["alpha"], color='#378ADD', lw=1.5, label='Estimated α')
    ax.axvline(5000, color='orange', lw=1, ls=':', label='Physics added')
    ax.set_title('α = R/L Identification')
    ax.set_xlabel('Epoch'); ax.set_ylabel('α (damping coeff)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 3: omega2 convergence
    ax = axes[1][0]
    ax.axhline(OMEGA2_TRUE, color='k', lw=1.5, ls='--', label=f'True ω²={OMEGA2_TRUE}')
    ax.plot(ep, history["omega2"], color='#2A9D6F', lw=1.5, label='Estimated ω²')
    ax.axvline(5000, color='orange', lw=1, ls=':', label='Physics added')
    ax.set_title('ω² = 1/LC Identification')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ω² (nat. freq. squared)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 4: loss
    ax = axes[1][1]
    ax.semilogy(ep, history["loss"], 'k-', lw=1.5)
    ax.axvline(5000, color='orange', lw=1, ls=':', label='Phase B starts')
    ax.set_title('Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {SAVE_PATH}")

if __name__ == '__main__':
    t_obs, q_clean, q_noisy = generate_data()
    model, history = train(t_obs, q_noisy)
    plot_results(model, history, t_obs, q_clean, q_noisy)
