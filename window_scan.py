import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
# ========== Basic settings ==========
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
r_eps = 1e-6
r_max = 15.0
# ========== Field equation ==========
def make_ode(m_eff_sq, beta, gamma):
    def ode_system(r, y):
        phi, dphi = y
        d2phi = -(2.0/r)*dphi + m_eff_sq*phi - beta*phi**3 - gamma*phi**5
        return [dphi, d2phi]
    return ode_system
# ========== Shooting method ==========
def target_residual(phi0, ode_system):
    sol = solve_ivp(ode_system, t_span=(r_eps, r_max), y0=[phi0, 0.0],
                    method="RK45", rtol=1e-8, atol=1e-10, max_step=0.02)
    if not sol.success:
        return 1e6
    return sol.y[0, -1]
def solve_shooting(m_eff_sq, beta, gamma, phi0_lo=0.5, phi0_hi=3.0, N_scan=80):
    ode_system = make_ode(m_eff_sq, beta, gamma)
    phi0_grid = np.linspace(phi0_lo, phi0_hi, N_scan)
    residuals = []
    for p in phi0_grid:
        try:
            residuals.append(target_residual(p, ode_system))
        except Exception:
            residuals.append(np.nan)
    residuals = np.array(residuals)
    brackets = []
    for i in range(N_scan-1):
        if np.isnan(residuals[i]) or np.isnan(residuals[i+1]):
            continue
        if residuals[i] * residuals[i+1] < 0:
            brackets.append((phi0_grid[i], phi0_grid[i+1]))
    if not brackets:
        return None
    a, b = min(brackets, key=lambda x: (x[0]+x[1])/2)
    try:
        phi0_root = brentq(lambda p: target_residual(p, ode_system), a, b, xtol=1e-10)
    except Exception:
        return None
    sol = solve_ivp(ode_system, t_span=(r_eps, r_max), y0=[phi0_root, 0.0],
                    method="RK45", rtol=1e-9, atol=1e-12, max_step=0.01,
                    dense_output=True)
    r = np.linspace(r_eps, r_max, 2000)
    y = sol.sol(r)
    phi = y[0, :]
    dphi = y[1, :]
    rho = 0.5*dphi**2 + 0.5*m_eff_sq*phi**2 - 0.25*beta*phi**4 - (1/6)*gamma*phi**6
    M = np.trapezoid(4*np.pi*r**2*rho, r)
    return phi0_root, M
# ========== Parameter window identification ==========
print("Parameter window identification: scanning m_eff_sq, beta, gamma")
print("Anchor: BESIII X(2370) = 2.395 +/- 0.011 GeV")
print("="*100)
m_eff_sq_grid = np.linspace(0.95, 1.25, 30)
beta_grid = np.linspace(7.0, 9.0, 30)
gamma_fixed = 0.10
M_grid = np.full((len(beta_grid), len(m_eff_sq_grid)), np.nan)
for i, beta in enumerate(beta_grid):
    for j, m_eff_sq in enumerate(m_eff_sq_grid):
        try:
            res = solve_shooting(m_eff_sq, beta, gamma_fixed)
            if res is not None:
                _, M = res
                M_grid[i, j] = M
        except Exception:
            pass
    if (i+1) % 5 == 0:
        print(f"  beta = {beta:.2f} done ({i+1}/{len(beta_grid)})")
print("="*100)
# ========== Window boundary identification ==========
print("\nBESIII window boundary (gamma = 0.10):")
print("="*70)
print(f"{'m_eff_sq':>10} | {'beta range':>22} | {'M range':>22}")
print("-"*70)
for j, m_eff_sq in enumerate(m_eff_sq_grid):
    col = M_grid[:, j]
    in_window = (col >= 2.384) & (col <= 2.406)
    if np.any(in_window):
        b_lo = beta_grid[in_window].min()
        b_hi = beta_grid[in_window].max()
        M_lo = col[in_window].min()
        M_hi = col[in_window].max()
        print(f"{m_eff_sq:>10.4f} | [{b_lo:.3f}, {b_hi:.3f}] | [{M_lo:.4f}, {M_hi:.4f}]")
# ========== Window central line fitting ==========
hit_beta = []
hit_m = []
for i, beta in enumerate(beta_grid):
    for j, m_eff_sq in enumerate(m_eff_sq_grid):
        if not np.isnan(M_grid[i, j]) and 2.384 <= M_grid[i, j] <= 2.406:
            hit_beta.append(beta)
            hit_m.append(m_eff_sq)
if hit_beta:
    print("\n" + "="*70)
    print("Overall boundary of the stable window:")
    print("="*70)
    print(f"  beta      in [{min(hit_beta):.3f}, {max(hit_beta):.3f}]")
    print(f"  m_eff_sq  in [{min(hit_m):.4f}, {max(hit_m):.4f}]")
    print(f"  Number of hit points: {len(hit_beta)}")
# ========== Parameter-mass heatmap ==========
fig, ax = plt.subplots(figsize=(11, 8))
vmin, vmax = 1.5, 3.5
im = ax.imshow(M_grid, origin='lower', aspect='auto',
               extent=[m_eff_sq_grid[0], m_eff_sq_grid[-1], beta_grid[0], beta_grid[-1]],
               cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
X, Y = np.meshgrid(m_eff_sq_grid, beta_grid)
cs = ax.contour(X, Y, M_grid, levels=[2.0, 2.4, 2.8], colors='white', linewidths=1.0)
ax.clabel(cs, inline=True, fontsize=10, fmt='%.1f')
cs_window = ax.contour(X, Y, M_grid, levels=[2.384, 2.406],
                       colors='black', linewidths=2.5, linestyles='solid')
ax.clabel(cs_window, inline=True, fontsize=11, fmt='%.3f')
window_mask = (M_grid >= 2.384) & (M_grid <= 2.406)
ax.contourf(X, Y, window_mask.astype(float), levels=[0.5, 1.5],
            colors=['black'], alpha=0.15)
ax.set_xlabel('m_eff_sq (GeV^2)', fontsize=13)
ax.set_ylabel('beta (GeV^-2)', fontsize=13)
ax.set_title('Parameter-mass map (gamma = 0.10)\nSolid black lines: BESIII window boundaries (2.384 and 2.406 GeV)', fontsize=14)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Soliton mass M (GeV)', fontsize=12)
plt.tight_layout()
plt.savefig('parameter_mass_map.png', dpi=150)
plt.show()
