"""
PINN for Series RLC Circuit
============================
Solves: L * q'' + R * q' + q/C = 0  (free oscillation)
Initial conditions: q(0) = 1, q'(0) = 0

Physics: Kirchhoff's Voltage Law (KVL)
Author: Built with Claude
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────
# 1. Circuit Parameters
# ─────────────────────────────────────────
R = 10.0    # Ohms
L = 1.0     # Henries
C = 0.01    # Farads
# Natural frequency: omega_0 = 1/sqrt(LC) ≈ 10 rad/s
# Damping ratio: zeta = R/(2*sqrt(L/C)) ≈ 0.5 (underdamped)

T_START = 0.0
T_END   = 2.0
N_COLLOCATION = 2000   # Physics residual points
N_BOUNDARY    = 1      # IC points

EPOCHS    = 15000
LR        = 1e-3
HIDDEN    = 64
LAYERS    = 4

# ─────────────────────────────────────────
# 2. Neural Network Architecture
# ─────────────────────────────────────────
class PINN(nn.Module):
    """
    Input:  t  (scalar time)
    Output: q(t) (charge on capacitor)
    Architecture: fully connected, Tanh activations
    """
    def __init__(self, hidden=HIDDEN, n_layers=LAYERS):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        # Xavier initialization for better convergence
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


# ─────────────────────────────────────────
# 3. Physics Loss (KVL Residual)
# ─────────────────────────────────────────
def physics_loss(model, t_phys):
    """
    Compute residual of: L*q'' + R*q' + q/C = 0
    Using automatic differentiation (autograd).
    """
    t_phys = t_phys.requires_grad_(True)
    q = model(t_phys)

    # First derivative: dq/dt (current)
    q_t = torch.autograd.grad(
        q, t_phys,
        grad_outputs=torch.ones_like(q),
        create_graph=True
    )[0]

    # Second derivative: d²q/dt²
    q_tt = torch.autograd.grad(
        q_t, t_phys,
        grad_outputs=torch.ones_like(q_t),
        create_graph=True
    )[0]

    # KVL residual: L*q'' + R*q' + q/C = 0
    residual = L * q_tt + R * q_t + q / C
    return torch.mean(residual ** 2)


# ─────────────────────────────────────────
# 4. Initial Condition Loss
# ─────────────────────────────────────────
def ic_loss(model):
    """
    Enforce: q(0) = 1 (initial charge)
             q'(0) = 0 (initially no current)
    """
    t0 = torch.tensor([[0.0]], requires_grad=True)
    q0 = model(t0)

    # q'(0)
    q0_t = torch.autograd.grad(
        q0, t0,
        grad_outputs=torch.ones_like(q0),
        create_graph=True
    )[0]

    loss_q  = (q0 - 1.0) ** 2
    loss_qt = (q0_t - 0.0) ** 2
    return loss_q + loss_qt


# ─────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────
def train():
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    # Collocation points (randomly sampled in [0, T])
    t_phys = torch.FloatTensor(N_COLLOCATION, 1).uniform_(T_START, T_END)

    history = {'total': [], 'physics': [], 'ic': []}

    print(f"Training PINN for RLC Circuit...")
    print(f"R={R}Ω, L={L}H, C={C}F")
    print(f"{'Epoch':>8} | {'Total':>12} | {'Physics':>12} | {'IC':>12}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        # Re-sample collocation points every 1000 epochs (curriculum)
        if epoch % 1000 == 0:
            t_phys = torch.FloatTensor(N_COLLOCATION, 1).uniform_(T_START, T_END)

        loss_phys = physics_loss(model, t_phys)
        loss_ic   = ic_loss(model)

        # Weighted loss: IC is critical for correctness
        loss = loss_phys + 100.0 * loss_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        history['total'].append(loss.item())
        history['physics'].append(loss_phys.item())
        history['ic'].append(loss_ic.item())

        if epoch % 1000 == 0:
            print(f"{epoch:>8} | {loss.item():>12.6f} | {loss_phys.item():>12.6f} | {loss_ic.item():>12.6f}")

    print("\nTraining complete!")
    return model, history


# ─────────────────────────────────────────
# 6. Analytical / ODE Reference Solution
# ─────────────────────────────────────────
def ode_reference():
    """Solve with scipy as ground truth."""
    def rlc_ode(t, y):
        q, dq = y
        d2q = (-R * dq - q / C) / L
        return [dq, d2q]

    t_eval = np.linspace(T_START, T_END, 500)
    sol = solve_ivp(rlc_ode, [T_START, T_END], [1.0, 0.0],
                    t_eval=t_eval, method='RK45', rtol=1e-8)
    return sol.t, sol.y[0]


# ─────────────────────────────────────────
# 7. Plotting
# ─────────────────────────────────────────
def plot_results(model, history):
    t_ref, q_ref = ode_reference()

    t_test = torch.linspace(T_START, T_END, 500).reshape(-1, 1)
    with torch.no_grad():
        q_pred = model(t_test).numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('PINN vs ODE Solver — Series RLC Circuit', fontsize=14, fontweight='bold')

    # --- Plot 1: Solution comparison ---
    ax1 = axes[0]
    ax1.plot(t_ref, q_ref, 'b-', linewidth=2.5, label='ODE Solver (ground truth)')
    ax1.plot(t_test.numpy(), q_pred, 'r--', linewidth=2, label='PINN Prediction')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Charge q(t) [C]', fontsize=12)
    ax1.set_title(f'R={R}Ω, L={L}H, C={C}F', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Compute error
    q_interp = np.interp(t_test.numpy().flatten(), t_ref, q_ref)
    max_err = np.max(np.abs(q_pred - q_interp))
    ax1.text(0.02, 0.05, f'Max error: {max_err:.4f} C',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Plot 2: Training loss ---
    ax2 = axes[1]
    epochs = range(1, EPOCHS + 1)
    ax2.semilogy(epochs, history['total'],   'k-',  linewidth=1.5, label='Total loss')
    ax2.semilogy(epochs, history['physics'], 'b--', linewidth=1.2, label='Physics loss')
    ax2.semilogy(epochs, history['ic'],      'r:',  linewidth=1.2, label='IC loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Training History', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pinn_rlc_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved to pinn_rlc_results.png")
    plt.show()


# ─────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(42)
    model, history = train()
    plot_results(model, history)

    # Save model
    torch.save(model.state_dict(), '/mnt/user-data/outputs/pinn_rlc_model.pth')
    print("Model saved to pinn_rlc_model.pth")
