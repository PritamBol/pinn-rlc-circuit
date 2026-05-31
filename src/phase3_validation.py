"""
PINN Phase 3 — Validation & Error Analysis
===========================================
Tests the PINN across three damping regimes:
  1. Underdamped  (R=10,  ζ≈0.5)  — oscillates
  2. Critical     (R=20,  ζ=1.0)  — fastest decay, no overshoot
  3. Overdamped   (R=40,  ζ=2.0)  — slow exponential decay

For each regime we:
  - Train a fresh PINN
  - Compare against scipy RK45 (ground truth)
  - Compute Max Absolute Error and L2 Relative Error
  - Plot results side by side
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────
# Circuit constants (fixed)
# ─────────────────────────────────────────
L = 1.0
C = 0.01
T_START, T_END = 0.0, 2.0
EPOCHS = 12000
N_COLL = 2000

# Three test cases
REGIMES = [
    {"name": "Underdamped",  "R": 10.0, "color": "#378ADD"},
    {"name": "Critical",     "R": 20.0, "color": "#2A9D6F"},
    {"name": "Overdamped",   "R": 40.0, "color": "#D85A30"},
]

# ─────────────────────────────────────────
# Neural Network (same as Phase 2)
# ─────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, hidden=64, n_layers=4):
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

    def forward(self, t):
        return self.net(t)


# ─────────────────────────────────────────
# Loss functions (parameterized by R)
# ─────────────────────────────────────────
def physics_loss(model, t_phys, R):
    t_phys = t_phys.requires_grad_(True)
    q = model(t_phys)
    q_t = torch.autograd.grad(q, t_phys,
              grad_outputs=torch.ones_like(q), create_graph=True)[0]
    q_tt = torch.autograd.grad(q_t, t_phys,
              grad_outputs=torch.ones_like(q_t), create_graph=True)[0]
    residual = L * q_tt + R * q_t + q / C
    return torch.mean(residual ** 2)

def ic_loss(model):
    t0 = torch.tensor([[0.0]], requires_grad=True)
    q0 = model(t0)
    q0_t = torch.autograd.grad(q0, t0,
               grad_outputs=torch.ones_like(q0), create_graph=True)[0]
    return (q0 - 1.0) ** 2 + q0_t ** 2


# ─────────────────────────────────────────
# Train one PINN for a given R
# ─────────────────────────────────────────
def train_one(R, seed=42):
    torch.manual_seed(seed)
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)
    t_phys = torch.FloatTensor(N_COLL, 1).uniform_(T_START, T_END)

    zeta = (R / 2) * (C / L) ** 0.5
    print(f"  Training: R={R}Ω  ζ={zeta:.3f}")

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        if epoch % 1000 == 0:
            t_phys = torch.FloatTensor(N_COLL, 1).uniform_(T_START, T_END)
        loss = physics_loss(model, t_phys, R) + 100.0 * ic_loss(model)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 4000 == 0:
            print(f"    Epoch {epoch:>6} | Loss: {loss.item():.6f}")
    return model


# ─────────────────────────────────────────
# Reference ODE solution (scipy RK45)
# ─────────────────────────────────────────
def ode_reference(R):
    def rlc(t, y):
        q, dq = y
        return [dq, (-R * dq - q / C) / L]
    t_eval = np.linspace(T_START, T_END, 500)
    sol = solve_ivp(rlc, [T_START, T_END], [1.0, 0.0],
                    t_eval=t_eval, method='RK45', rtol=1e-9)
    return sol.t, sol.y[0]


# ─────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────
def compute_errors(q_pred, q_ref):
    """
    Max Absolute Error  : worst-case pointwise error
    L2 Relative Error   : overall error normalized by signal energy
    """
    max_err = np.max(np.abs(q_pred - q_ref))
    l2_err  = np.linalg.norm(q_pred - q_ref) / (np.linalg.norm(q_ref) + 1e-10)
    return max_err, l2_err


# ─────────────────────────────────────────
# Main: train all three, plot, compare
# ─────────────────────────────────────────
def main():
    t_test = torch.linspace(T_START, T_END, 500).reshape(-1, 1)
    t_np   = t_test.numpy().flatten()

    results = []

    for reg in REGIMES:
        R    = reg["R"]
        name = reg["name"]
        zeta = (R / 2) * (C / L) ** 0.5

        print(f"\n{'='*50}")
        print(f"Regime: {name}  (R={R}Ω, ζ={zeta:.3f})")
        print('='*50)

        model = train_one(R)

        # PINN prediction
        with torch.no_grad():
            q_pred = model(t_test).numpy().flatten()

        # Reference solution (interpolated to same grid)
        t_ref, q_ref_raw = ode_reference(R)
        q_ref = np.interp(t_np, t_ref, q_ref_raw)

        # Errors
        max_err, l2_err = compute_errors(q_pred, q_ref)
        print(f"  Max Absolute Error : {max_err:.6f} C")
        print(f"  L2 Relative Error  : {l2_err*100:.4f} %")

        results.append({
            "name": name, "R": R, "zeta": zeta,
            "color": reg["color"],
            "q_pred": q_pred, "q_ref": q_ref,
            "max_err": max_err, "l2_err": l2_err
        })

    # ─────────────────────────────────────
    # Plot: 3 rows × 2 columns
    # Left column:  PINN vs ODE
    # Right column: pointwise absolute error
    # ─────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle('Phase 3 — PINN Validation Across Damping Regimes',
                 fontsize=14, fontweight='bold')

    for i, res in enumerate(results):
        t_np_plot = t_test.numpy().flatten()
        error = np.abs(res["q_pred"] - res["q_ref"])

        # Left: solution comparison
        ax = axes[i][0]
        ax.plot(t_np_plot, res["q_ref"], 'k-', lw=2.5, label='ODE Solver (truth)')
        ax.plot(t_np_plot, res["q_pred"], '--', color=res["color"], lw=2, label='PINN')
        ax.set_title(f'{res["name"]}  R={res["R"]}Ω  ζ={res["zeta"]:.2f}')
        ax.set_ylabel('q(t) [C]')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.05,
                f'Max err: {res["max_err"]:.5f} C\nL2 err: {res["l2_err"]*100:.3f}%',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Right: pointwise error
        ax2 = axes[i][1]
        ax2.fill_between(t_np_plot, error, alpha=0.35, color=res["color"])
        ax2.plot(t_np_plot, error, color=res["color"], lw=1.5)
        ax2.set_title(f'{res["name"]} — Pointwise Absolute Error')
        ax2.set_ylabel('|q_pred − q_ref| [C]')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pinn_phase3_validation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: pinn_phase3_validation.png")
    plt.show()

    # Summary table
    print("\n" + "="*55)
    print(f"{'Regime':<14} | {'R (Ω)':>6} | {'ζ':>6} | {'Max Err':>12} | {'L2 Err %':>10}")
    print("-"*55)
    for r in results:
        print(f"{r['name']:<14} | {r['R']:>6.1f} | {r['zeta']:>6.2f} | "
              f"{r['max_err']:>12.6f} | {r['l2_err']*100:>10.4f}")
    print("="*55)


if __name__ == '__main__':
    main()
