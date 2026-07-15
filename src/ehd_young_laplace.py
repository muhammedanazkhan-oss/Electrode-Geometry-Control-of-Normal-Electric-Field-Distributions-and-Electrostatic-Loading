"""
ehd_young_laplace.py -- Equilibrium sessile-droplet shapes from the augmented
Young-Laplace equation including hydrostatic pressure (Reviewer 4, comment 2).

The axisymmetric interface is parameterised by arc length s from the apex:
    dx/ds   = cos(psi)
    dz/ds   = sin(psi)
    dpsi/ds = 2/b + (rho g / gamma) * z - sin(psi)/x      (x>0)
with the apex regularisation dpsi/ds -> 1/b as x->0.  Here b is the apex curvature
radius and psi the local tangent angle.  Integration proceeds from the apex until
psi reaches the prescribed contact angle theta_c; the drop then has footprint radius
x_c and height z_c.  b is chosen (shooting) so that the enclosed volume equals a
target, allowing shapes of equal volume but different wettability/gravity to be
compared on an equal-material basis.

Bond number Bo = rho g R^2 / gamma; capillary length l_c = sqrt(gamma/(rho g)).
"""
import numpy as np
from scipy.integrate import solve_ivp

def shape(b, theta_c, gamma, rho, g, n=4000):
    cap = rho * g / gamma
    def rhs(s, y):
        x, z, psi = y
        dpsi = (2.0 / b + cap * z - (np.sin(psi) / x if x > 1e-9 else 0.0))
        if x < 1e-9:
            dpsi = 1.0 / b
        return [np.cos(psi), np.sin(psi), dpsi]
    def hit(s, y): return y[2] - theta_c
    hit.terminal = True; hit.direction = 1
    sol = solve_ivp(rhs, [0, 20e-3], [1e-7, 0.0, 1e-7], events=hit,
                    max_step=5e-6, rtol=1e-8, atol=1e-10, dense_output=True)
    x, z, psi = sol.y
    return x, z, psi

def volume(x, z):
    # solid of revolution, disc integration: V = int pi x^2 dz  (z from apex down)
    return np.abs(np.trapz(np.pi * x**2, z))

def solve_shape_for_volume(theta_c, V_target, gamma, rho, g, b0=4e-3):
    from scipy.optimize import brentq
    def f(b):
        x, z, _ = shape(b, theta_c, gamma, rho, g)
        return volume(x, z) - V_target
    b = brentq(f, 0.3e-3, 20e-3, xtol=1e-7)
    x, z, psi = shape(b, theta_c, gamma, rho, g)
    return b, x, z

if __name__ == '__main__':
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ehd_config import GAMMA, RHO_L, G, R_DROP
    mmv=1e-3
    Bo = RHO_L*G*R_DROP**2/GAMMA
    lc = np.sqrt(GAMMA/(RHO_L*G))
    print(f"R=4mm water drop:  Bond number Bo = {Bo:.3f},  capillary length l_c = {lc/mmv:.3f} mm")
    Vhemi = (2/3)*np.pi*R_DROP**3
    print(f"hemisphere volume V = {Vhemi*1e9:.1f} uL")
    # zero-gravity hemisphere reference vs gravity-equilibrated equal-volume shapes
    for th_deg in [70,90,110,130]:
        b,x,z = solve_shape_for_volume(np.deg2rad(th_deg), Vhemi, GAMMA, RHO_L, G)
        print(f"  theta_c={th_deg:3d}deg: apex_b={b/mmv:.3f}mm  footprint={x[-1]/mmv:.3f}mm  height={z[-1]/mmv:.3f}mm")
