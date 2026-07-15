"""
ehd_config.py -- The manuscript's electrode configurations, in one place.

Geometry (unchanged from the original submission unless stated):
  droplet        sessile water cap, curvature radius R = 4 mm, on the substrate z = 0
  bottom elec.   substrate plane z <= 0, held at +V (the droplet is in contact and
                 therefore at the same potential)
  PP top plate   26 mm wide, 1.5 mm thick, lower face at H = 10 mm, held at 0
  PT pin         1 mm diameter (a = 0.5 mm), 5 mm long, tip at z = 10 mm, held at 0,
                 lateral offset d from the droplet axis
  DP pins        two pins as above at x = +/-6 mm, at +20 kV and -20 kV, substrate at 0

Change introduced in revision: the pin tip carries a finite edge radius r_tip
(default 0.10 mm).  The original flat-ended, sharp-edged pin is available via
sharp=True for the explicit comparison requested by Reviewer 4 (comment 7).
"""
import numpy as np
from ehd_geom import (SphericalCapDroplet, RoundedPin, FlatPin, Plate, GroundPlane, Disc)

mm = 1e-3

# -- physical constants and liquid properties (deionised water at 25 C) -------
EPS0 = 8.8541878128e-12       # F/m
GAMMA = 7.28e-2               # N/m, surface tension
RHO_L = 997.0                 # kg/m^3
G = 9.81                      # m/s^2
EPS_R = 80.0
SIGMA_L = 5e-4                # S/m

# -- default geometry ---------------------------------------------------------
R_DROP = 4.0 * mm
H_GAP = 10.0 * mm
PIN_A = 0.5 * mm
PIN_LEN = 5.0 * mm
R_TIP = 0.10 * mm
PLATE_HW = 13.0 * mm
PLATE_T = 1.5 * mm

# -- Taylor criterion ---------------------------------------------------------
K_T = 4.47e5                  # V m^1/2 N^-1/2  (Taylor 1964)

def E_taylor(R=R_DROP, gamma=GAMMA):
    """Adopted local Taylor-field criterion E_T = K_T sqrt(gamma/R)."""
    return K_T * np.sqrt(gamma / R)

def E_rayleigh(R=R_DROP, gamma=GAMMA):
    """Isolated-sphere Rayleigh / pressure-balance reference, 2 sqrt(gamma/(eps0 R))."""
    return 2.0 * np.sqrt(gamma / (EPS0 * R))


# -- configuration builders ---------------------------------------------------
def make_droplet(V, R=R_DROP, theta_c=np.pi / 2):
    return SphericalCapDroplet(R, theta_c, potential=V)

def config_pp(V, R=R_DROP, theta_c=np.pi / 2):
    drop = make_droplet(V, R, theta_c)
    return [GroundPlane(0.0, potential=V), drop,
            Plate(PLATE_HW, H_GAP, H_GAP + PLATE_T, potential=0.0)], drop

def config_pin(V, d=0.0, R=R_DROP, theta_c=np.pi / 2, r_tip=R_TIP, sharp=False):
    drop = make_droplet(V, R, theta_c)
    if sharp:
        pin = FlatPin(d, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, potential=0.0)
    else:
        pin = RoundedPin(d, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, r_tip, potential=0.0)
    return [GroundPlane(0.0, potential=V), drop, pin], drop

def config_double_pin(Vp=20e3, x_off=6.0 * mm, R=R_DROP, theta_c=np.pi / 2,
                      r_tip=R_TIP, sharp=False):
    """Substrate and droplet grounded; the two pins at +Vp and -Vp."""
    drop = make_droplet(0.0, R, theta_c)
    P = FlatPin if sharp else RoundedPin
    kw = {} if sharp else {'r_tip': r_tip}
    pin_p = P(+x_off, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, potential=+Vp, **kw)
    pin_m = P(-x_off, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, potential=-Vp, **kw)
    return [GroundPlane(0.0, potential=0.0), drop, pin_p, pin_m], drop


# ============================================================================ #
#  Free-space (physically correct) configurations
#  ------------------------------------------------------------------------
#  The bottom electrode is the ACTUAL 20 mm-diameter, 2 mm-thick copper disc,
#  the droplet is in contact with it (same potential +V), the upper electrode
#  (plate or pin) is grounded, and phi -> 0 far away (earth at infinity),
#  approximated by phi = 0 on a large outer boundary whose size-independence is
#  verified explicitly.  This replaces the original infinite-biased-plane model,
#  which is not a well-posed free-space problem for the pin cell and leaves the
#  droplet-surface field dependent on the placement of the outer boundary.
# ============================================================================ #
DISC_R = 10.0 * mm
DISC_T = 2.0 * mm

def fs_pp(V, R=R_DROP, theta_c=np.pi / 2):
    drop = SphericalCapDroplet(R, theta_c, potential=V)
    return [Disc(DISC_R, -DISC_T, 0.0, potential=V), drop,
            Plate(PLATE_HW, H_GAP, H_GAP + PLATE_T, potential=0.0)], drop

def fs_pin(V, d=0.0, R=R_DROP, theta_c=np.pi / 2, r_tip=R_TIP, sharp=False):
    drop = SphericalCapDroplet(R, theta_c, potential=V)
    P = FlatPin(d, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, potential=0.0) if sharp \
        else RoundedPin(d, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, r_tip, potential=0.0)
    return [Disc(DISC_R, -DISC_T, 0.0, potential=V), drop, P], drop

def fs_bare(V, R=R_DROP, theta_c=np.pi / 2):
    drop = SphericalCapDroplet(R, theta_c, potential=V)
    return [Disc(DISC_R, -DISC_T, 0.0, potential=V), drop], drop

def fs_double_pin(Vp=20e3, x_off=6.0 * mm, R=R_DROP, theta_c=np.pi / 2,
                  r_tip=R_TIP, sharp=False):
    drop = SphericalCapDroplet(R, theta_c, potential=0.0)
    P = FlatPin if sharp else RoundedPin
    kw = {} if sharp else {'r_tip': r_tip}
    pp_ = P(+x_off, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, potential=+Vp, **kw)
    pm_ = P(-x_off, 0.0, PIN_A, H_GAP, H_GAP + PIN_LEN, potential=-Vp, **kw)
    return [Disc(DISC_R, -DISC_T, 0.0, potential=0.0), drop, pp_, pm_], drop
