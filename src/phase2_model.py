"""
PINN Phase 2 — Fully Annotated for Beginners
==============================================
We build a Physics-Informed Neural Network (PINN) to solve the
Series RLC circuit ODE:

    L * q''(t) + R * q'(t) + q(t)/C = 0

Initial conditions:
    q(0)  = 1   (initial charge on capacitor)
    q'(0) = 0   (no current at t=0)

We want the network to learn q(t) purely by satisfying this equation.
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import torch          # PyTorch — the deep learning library
import torch.nn as nn # Neural network building blocks
import numpy as np    # Numerical arrays (for plotting)
import matplotlib.pyplot as plt  # For plotting results
from scipy.integrate import solve_ivp  # Classical ODE solver (for comparison)


# ─────────────────────────────────────────────────────────────
# SECTION 1: CIRCUIT PARAMETERS
# ─────────────────────────────────────────────────────────────
# These are the physical values of the components.
# Changing them changes the behavior of the circuit.

R = 10.0    # Resistance in Ohms      — controls damping (energy loss)
L = 1.0     # Inductance in Henries   — controls oscillation inertia
C = 0.01    # Capacitance in Farads   — stores electric charge

# Time domain: we want to predict q(t) from t=0 to t=2 seconds
T_START = 0.0
T_END   = 2.0

# Quick sanity check: compute damping ratio
# zeta < 1 → underdamped (oscillates)
# zeta = 1 → critically damped
# zeta > 1 → overdamped (no oscillation)
zeta = (R / 2) * (C / L) ** 0.5
omega0 = 1.0 / (L * C) ** 0.5
print(f"Natural frequency ω₀ = {omega0:.2f} rad/s")
print(f"Damping ratio ζ = {zeta:.3f} → {'Underdamped' if zeta < 1 else 'Overdamped'}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: THE NEURAL NETWORK
# ─────────────────────────────────────────────────────────────
# The network maps:  t (scalar time) → q̂(t) (predicted charge)
#
# Architecture:
#   Input layer:   1 neuron  (time t)
#   Hidden layers: 4 × 64 neurons with Tanh activation
#   Output layer:  1 neuron  (charge q)
#
# WHY Tanh (not ReLU)?
#   We need to compute q'' (second derivative of network output).
#   ReLU's second derivative is ZERO everywhere → useless for physics.
#   Tanh is smooth and infinitely differentiable → perfect for PINNs.

class PINN(nn.Module):

    def __init__(self, hidden=64, n_layers=4):
        super().__init__()  # Always call this first — PyTorch requires it

        # Build layers programmatically
        # First layer: 1 input → 64 hidden neurons
        layers = [nn.Linear(1, hidden), nn.Tanh()]

        # Middle layers: 64 → 64 (repeated n_layers-1 times)
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]

        # Final layer: 64 → 1 output (no activation — free to output any number)
        layers.append(nn.Linear(hidden, 1))

        # nn.Sequential chains all layers: input flows through each one in order
        self.net = nn.Sequential(*layers)

        # Xavier initialization: sets initial weights to a good scale
        # Without this, gradients can vanish or explode in early training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        # Called when you do: model(t)
        # Just pass t through all the layers
        return self.net(t)


# ─────────────────────────────────────────────────────────────
# SECTION 3: PHYSICS LOSS
# ─────────────────────────────────────────────────────────────
# This is the KEY idea of PINN.
#
# We compute the residual of the RLC ODE:
#   residual = L*q'' + R*q' + q/C
#
# If the network perfectly satisfies the physics, residual = 0 everywhere.
# We minimize the mean squared residual.
#
# HOW do we get q' and q''?
#   NOT by finite differences. Instead, we use PyTorch's autograd.
#   autograd differentiates through the neural network analytically,
#   the same way it computes gradients for backpropagation.

def physics_loss(model, t_phys):
    """
    t_phys: tensor of shape (N, 1) — N collocation points in [0, T]
    Returns: scalar loss value
    """

    # STEP 1: Enable gradient tracking on input t
    # This tells PyTorch: "build a computation graph as the
    # forward pass runs, so we can differentiate w.r.t. t later"
    t_phys = t_phys.requires_grad_(True)

    # STEP 2: Forward pass — get network's prediction of q at each t
    q = model(t_phys)   # shape: (N, 1)

    # STEP 3: Compute dq/dt using autograd
    # torch.autograd.grad differentiates `q` with respect to `t_phys`
    # grad_outputs=ones tells it to sum gradients across the batch
    # create_graph=True keeps the graph so we can differentiate AGAIN
    q_t = torch.autograd.grad(
        outputs=q,
        inputs=t_phys,
        grad_outputs=torch.ones_like(q),
        create_graph=True   # <-- MUST be True so we can get q''
    )[0]   # returns a tuple; [0] gets the gradient tensor

    # STEP 4: Compute d²q/dt² — differentiate q' with respect to t again
    q_tt = torch.autograd.grad(
        outputs=q_t,
        inputs=t_phys,
        grad_outputs=torch.ones_like(q_t),
        create_graph=True
    )[0]

    # STEP 5: Compute ODE residual
    # This is the KVL equation: L*q'' + R*q' + q/C = 0
    # If network satisfies physics perfectly, residual = 0 at every point
    residual = L * q_tt + R * q_t + q / C

    # STEP 6: Mean squared residual — this is the loss we minimize
    return torch.mean(residual ** 2)


# ─────────────────────────────────────────────────────────────
# SECTION 4: INITIAL CONDITION LOSS
# ─────────────────────────────────────────────────────────────
# The physics loss alone has infinitely many solutions.
# (Any time-shifted or amplitude-scaled version also satisfies the ODE)
# Initial conditions pin down the SPECIFIC solution we want.
#
# We enforce:
#   q(0)  = 1    ← capacitor starts fully charged
#   q'(0) = 0    ← no current flowing at t=0

def ic_loss(model):
    """
    Returns: scalar loss penalizing violations of initial conditions
    """

    # Evaluate network exactly at t=0
    t0 = torch.tensor([[0.0]], requires_grad=True)
    q0 = model(t0)   # predicted q(0)

    # Compute q'(0) — initial current
    q0_t = torch.autograd.grad(
        outputs=q0,
        inputs=t0,
        grad_outputs=torch.ones_like(q0),
        create_graph=True
    )[0]

    # Squared error from desired initial values
    loss_position = (q0 - 1.0) ** 2   # q(0) should be 1
    loss_velocity = (q0_t - 0.0) ** 2 # q'(0) should be 0

    return loss_position + loss_velocity


# ─────────────────────────────────────────────────────────────
# SECTION 5: TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train():
    # Create the network
    model = PINN(hidden=64, n_layers=4)
    print(f"\nNetwork parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Adam optimizer — adapts learning rate per parameter automatically
    # lr=1e-3 is a good starting point for PINNs
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler: halve lr every 5000 epochs
    # This helps fine-tune the solution in later training
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5000, gamma=0.5
    )

    # COLLOCATION POINTS: random times in [T_START, T_END]
    # The physics loss is evaluated at these points.
    # Think of them as "checkpoints" where we verify the ODE is satisfied.
    # More points = better coverage = more accurate solution.
    N_COLLOCATION = 2000
    t_phys = torch.FloatTensor(N_COLLOCATION, 1).uniform_(T_START, T_END)

    # Track loss history for plotting
    history = []

    EPOCHS = 15000
    print(f"\n{'Epoch':>8} | {'Total Loss':>12} | {'Physics':>12} | {'IC':>12} | {'LR':>10}")
    print("-" * 62)

    for epoch in range(1, EPOCHS + 1):

        # ALWAYS zero gradients at the start of each iteration
        # PyTorch accumulates gradients by default — we don't want that
        optimizer.zero_grad()

        # Resample collocation points periodically
        # This prevents the network from overfitting to specific time locations
        if epoch % 1000 == 0:
            t_phys = torch.FloatTensor(N_COLLOCATION, 1).uniform_(T_START, T_END)

        # Compute both loss terms
        loss_phys = physics_loss(model, t_phys)
        loss_ic   = ic_loss(model)

        # TOTAL LOSS: weighted sum
        # WHY weight IC by 100?
        #   Physics loss averages over 2000 points → numerically small
        #   IC loss covers only 1 point → naturally small number
        #   Without weighting, optimizer ignores IC → wrong solution!
        loss = loss_phys + 100.0 * loss_ic

        # Backpropagation: compute gradients of loss w.r.t. all weights
        loss.backward()

        # Gradient step: update weights in direction that reduces loss
        optimizer.step()
        scheduler.step()

        history.append(loss.item())

        if epoch % 1000 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"{epoch:>8} | {loss.item():>12.6f} | "
                  f"{loss_phys.item():>12.6f} | {loss_ic.item():>12.6f} | {lr:>10.6f}")

    print("\nTraining complete!")
    return model, history


# ─────────────────────────────────────────────────────────────
# SECTION 6: REFERENCE SOLUTION (scipy)
# ─────────────────────────────────────────────────────────────
# We use scipy's RK45 solver as ground truth to validate PINN accuracy.
# This is a classical numerical solver — highly accurate but requires
# a mesh (discrete time steps). PINN is mesh-free.

def ode_reference():
    def rlc_ode(t, y):
        q, dq = y
        d2q = (-R * dq - q / C) / L
        return [dq, d2q]

    t_eval = np.linspace(T_START, T_END, 500)
    sol = solve_ivp(
        rlc_ode,
        [T_START, T_END],
        [1.0, 0.0],        # initial conditions: q(0)=1, q'(0)=0
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8
    )
    return sol.t, sol.y[0]


# ─────────────────────────────────────────────────────────────
# SECTION 7: PLOT RESULTS
# ─────────────────────────────────────────────────────────────

def plot_results(model, history):
    # Get reference solution
    t_ref, q_ref = ode_reference()

    # Get PINN prediction on a fine grid
    t_test = torch.linspace(T_START, T_END, 500).reshape(-1, 1)
    with torch.no_grad():  # No need to track gradients for inference
        q_pred = model(t_test).numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('PINN Phase 2 — Series RLC Circuit', fontsize=14, fontweight='bold')

    # Plot 1: Solution comparison
    ax1 = axes[0]
    ax1.plot(t_ref, q_ref, 'b-', linewidth=2.5, label='ODE Solver (ground truth)')
    ax1.plot(t_test.numpy(), q_pred, 'r--', linewidth=2, label='PINN Prediction')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Charge q(t) [C]')
    ax1.set_title(f'R={R}Ω  L={L}H  C={C}F  |  ζ={zeta:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Max error
    q_interp = np.interp(t_test.numpy().flatten(), t_ref, q_ref)
    max_err = np.max(np.abs(q_pred - q_interp))
    ax1.text(0.02, 0.05, f'Max error: {max_err:.5f} C',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Training loss
    ax2 = axes[1]
    ax2.semilogy(history, 'k-', linewidth=1.5, label='Total loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pinn_phase2_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved!")
    plt.show()


# ─────────────────────────────────────────────────────────────
# SECTION 8: RUN EVERYTHING
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)  # For reproducibility
    model, history = train()
    plot_results(model, history)
    torch.save(model.state_dict(), '/mnt/user-data/outputs/pinn_phase2_model.pth')
    print("Model weights saved!")
