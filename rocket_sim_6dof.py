#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 6-DOF ROCKET FLIGHT DYNAMICS SIMULATOR
 Single-file, physics-rigorous ascent / coast / ballistic-return simulation
 with GNC, Monte Carlo dispersion analysis and trajectory optimization.
================================================================================

 STATE VECTOR (13 rigid-body states + 2 TVC actuator states)
 ----------------------------------------------------------------------------
   y[0:3]   r_I      [m]      Position in the Earth-Centered Inertial frame
   y[3:6]   v_I      [m/s]    Inertial velocity
   y[6:10]  q        [-]      Quaternion, BODY -> INERTIAL, scalar first
                              q = [q0, q1, q2, q3],  |q| = 1
   y[10:13] w_B      [rad/s]  Body angular rate  [p, q, r] (roll, pitch, yaw)
   y[13:15] delta    [rad]    TVC gimbal angles  [delta_p, delta_y]
                              (actuator states: slew-rate + lag limited)

 FRAMES & CONVENTIONS
 ----------------------------------------------------------------------------
   ECI   : Earth-centered, non-rotating (planet rotation neglected -> ECI==ECEF;
           a documented modelling assumption appropriate for a short suborbital
           flight; gravity is the central inverse-square field of the spec,
           g(h) = g0 * (Re/(Re+h))^2 ).
   NED   : local North-East-Down frame anchored at the launch pad; used for
           guidance references, Euler-angle telemetry and dispersion plots.
   BODY  : x_B out the nose (roll axis), y_B right, z_B completing the RH set.
           Longitudinal stations (CG, CP, nozzle) are measured FROM THE NOSE,
           positive aft; a station s sits at body coordinates [-s, 0, 0].
           (The spec's "z_cg along the longitudinal axis" is this station
           coordinate; here the longitudinal axis is body-x per the standard
           p-q-r convention.)

 EQUATIONS OF MOTION (fully expanded)
 ----------------------------------------------------------------------------
   r_dot   = v
   v_dot   = R_IB(q) * F_B / m  +  g(r)                       [translation]
   q_dot   = 0.5 * q (x) [0, w_B]  +  k_q (1 - |q|^2) q       [attitude]
             (quaternion kinematics + Baumgarte-style normalization feedback;
              q is additionally re-normalized at every derivative evaluation)
   w_dot   = I^-1 [ M_aero + M_thrust + M_jet
                    - w x (I w) - I_dot w ]                    [rotation]
             with the variable-mass terms:
               I_dot w  : instantaneous inertia-rate torque (I(t) from CG/prop
                          motion, evaluated by central finite difference)
               M_jet    = - mdot * l_n^2 * [0, q, r]
                          jet damping: expelled propellant leaves through the
                          nozzle at lever arm l_n and carries away transverse
                          angular momentum (net stabilizing, Thomson Ch. 8).

 REQUIREMENTS TRACE (spec bullet -> implementation)
 ----------------------------------------------------------------------------
   State vector / quaternion kinematics ......... dynamics(), quat_* helpers
   Frame transformations (I/B/NED) .............. quat_to_dcm, LaunchFrame
   Variable mass, CG shift, I(t) ................ Vehicle.mass_properties
   US Standard Atmosphere 1976 (<=86 km) ........ atmosphere()
   g(h) = g0 (Re/(Re+h))^2 ...................... gravity_eci()
   Altitude-dependent crosswind ................. wind at _forces_body()
   CD(M), CL(alpha), CY(beta) ................... aero block in _forces_body()
       (implemented as a body-of-revolution total-angle-of-attack model:
        normal force N = Q S CNa sin(a_t) applied opposite the cross-flow
        unit vector; for small angles this reduces exactly to
        CL = CNa*alpha and CY = -CNa*beta, and remains valid while tumbling)
   Forces/moments at CP, CP-vs-CG travel ........ x_cp(M) table, lever arms
   Damping moments Clp, Cmq, Cnr ................ damping block
   2-axis TVC w/ limits & slew rates ............ actuator states y[13:15]
   RK45 adaptive integration .................... simulate() via solve_ivp
   Quaternion normalization each step ........... dynamics() + k_q feedback
   Vehicle definition struct .................... Vehicle dataclass
   Pitchover + gravity-turn guidance ............ guidance_direction()
   PD attitude control -> gimbal ................ _tvc_command()
   Monte Carlo (N=100, spec dispersions) ........ run_monte_carlo()
   Trajectory optimization (insertion velocity
     objective, max-Q constrained -- see note) ... optimize_pitchover()
   Visualization suite .......................... plot_* functions

 RUN:   python rocket_sim_6dof.py
        (renders and saves: rocketsim_trajectory3d.png,
         rocketsim_dashboard.png, rocketsim_montecarlo.png)

 Dependencies: numpy, scipy, matplotlib (standard scientific stack only).
================================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ==============================================================================
# SECTION 0 -- PHYSICAL CONSTANTS AND GLOBAL CONFIGURATION
# ==============================================================================

G0        = 9.80665          # standard gravity                        [m/s^2]
R_EARTH   = 6_371_000.0      # mean spherical Earth radius             [m]
R0_GEOPOT = 6_356_766.0      # USSA-76 geopotential reference radius   [m]
R_AIR     = 287.05287        # specific gas constant of air            [J/kg/K]
GAMMA_AIR = 1.4              # ratio of specific heats                 [-]
P0_SL     = 101_325.0        # sea-level standard pressure             [Pa]
T0_SL     = 288.15           # sea-level standard temperature          [K]

N_MONTE_CARLO = 100          # number of stochastic dispersion runs
RNG_SEED      = 20260818     # reproducibility seed for the MC engine
Q_LIMIT_PA    = 45_000.0     # ascent max-Q constraint (optimizer)     [Pa]

FIG_DPI = 150

# Validated categorical palette + chart chrome (see module docstring notes).
COL = {
    "s1":      "#2a78d6",    # series 1 (nominal)        - blue
    "s2":      "#eb6834",    # series 2 (optimized)      - orange
    "s3":      "#1baf7a",    # series 3                  - aqua
    "s4":      "#eda100",    # series 4                  - yellow
    "crit":    "#d03b3b",    # constraint / limit lines  - status-critical
    "ink":     "#0b0b0b",
    "sub":     "#52514e",
    "muted":   "#898781",
    "grid":    "#e1e0d9",
    "axis":    "#c3c2b7",
    "surface": "#fcfcfb",
}

# ==============================================================================
# SECTION 1 -- QUATERNION AND FRAME UTILITIES
# ==============================================================================

def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3-vector cross product (explicit; faster than np.cross for scalars)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def quat_mult(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product q (x) p, scalar-first convention."""
    q0, q1, q2, q3 = q
    p0, p1, p2, p3 = p
    return np.array([
        q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3,
        q0 * p1 + q1 * p0 + q2 * p3 - q3 * p2,
        q0 * p2 - q1 * p3 + q2 * p0 + q3 * p1,
        q0 * p3 + q1 * p2 - q2 * p1 + q3 * p0,
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm (identity quaternion if degenerate)."""
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Direction cosine matrix R_IB for a unit quaternion (scalar first):
    transforms BODY-frame vectors into the INERTIAL frame,  v_I = R_IB @ v_B.
    """
    q0, q1, q2, q3 = q
    return np.array([
        [1.0 - 2.0 * (q2 * q2 + q3 * q3), 2.0 * (q1 * q2 - q0 * q3),       2.0 * (q1 * q3 + q0 * q2)],
        [2.0 * (q1 * q2 + q0 * q3),       1.0 - 2.0 * (q1 * q1 + q3 * q3), 2.0 * (q2 * q3 - q0 * q1)],
        [2.0 * (q1 * q3 - q0 * q2),       2.0 * (q2 * q3 + q0 * q1),       1.0 - 2.0 * (q1 * q1 + q2 * q2)],
    ])


def dcm_to_quat(R: np.ndarray) -> np.ndarray:
    """Robust DCM -> quaternion (Shepperd's method, scalar first)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s,
                      0.25 * s,
                      (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s,
                      (R[0, 1] + R[1, 0]) / s,
                      0.25 * s,
                      (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s,
                      (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s,
                      0.25 * s])
    return quat_normalize(q)


def euler_zyx_from_dcm(R_nb: np.ndarray) -> tuple[float, float, float]:
    """
    3-2-1 (yaw-pitch-roll) Euler angles from a BODY->NED DCM.
    Returns (roll, pitch, yaw) [rad]. Display-only quantity: the quaternion
    remains the attitude state, so the pitch ~ +-90 deg extraction singularity
    affects telemetry cosmetics only, never the dynamics.
    """
    pitch = -math.asin(max(-1.0, min(1.0, R_nb[2, 0])))
    roll  = math.atan2(R_nb[2, 1], R_nb[2, 2])
    yaw   = math.atan2(R_nb[1, 0], R_nb[0, 0])
    return roll, pitch, yaw

# ==============================================================================
# SECTION 2 -- ENVIRONMENT: US STANDARD ATMOSPHERE 1976 AND GRAVITY
# ==============================================================================
# Seven-layer USSA-76 up to 86 km geometric altitude. Base pressures are
# integrated at import time from the defining lapse rates, which reproduces
# the published tables (verified in _self_test to <0.3%):
#   H = 11 km : T = 216.65 K, P = 22632.06 Pa
#   H = 20 km : P = 5474.889 Pa      H = 32 km : P = 868.0187 Pa
#   H = 47 km : T = 270.65 K, P = 110.9063 Pa

_H_BASE = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0,
                    51_000.0, 71_000.0, 84_852.0])          # geopotential [m']
_LAPSE  = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0,
                    -0.0028, -0.0020])                       # [K/m']


def _build_atmosphere_bases() -> tuple[np.ndarray, np.ndarray]:
    """Integrate the hydrostatic relation layer-by-layer for base T and P."""
    Tb, Pb = [T0_SL], [P0_SL]
    for i in range(len(_LAPSE)):
        dH, L = _H_BASE[i + 1] - _H_BASE[i], _LAPSE[i]
        T, P = Tb[i], Pb[i]
        T_next = T + L * dH
        if L == 0.0:                                # isothermal layer
            P_next = P * math.exp(-G0 * dH / (R_AIR * T))
        else:                                       # gradient layer
            P_next = P * (T_next / T) ** (-G0 / (R_AIR * L))
        Tb.append(T_next)
        Pb.append(P_next)
    return np.array(Tb), np.array(Pb)


_T_BASE, _P_BASE = _build_atmosphere_bases()


def atmosphere(h_geometric: float) -> tuple[float, float, float, float]:
    """
    US Standard Atmosphere 1976.

    Parameters:  h_geometric [m] geometric altitude above the spherical MSL.
    Returns:     (T [K], P [Pa], rho [kg/m^3], a [m/s]).

    Below 0 m the sea-level state is returned (impact-event bracketing);
    above 86 km geometric an isothermal exponential continuation of the 86-km
    state is used (density there is < 1e-5 of sea level, so the approximation
    has no dynamical significance -- documented model boundary).
    """
    h = max(h_geometric, 0.0)
    H = R0_GEOPOT * h / (R0_GEOPOT + h)             # geometric -> geopotential

    if H >= _H_BASE[-1]:
        T = _T_BASE[-1]                             # ~186.9 K, isothermal cap
        rho_base = _P_BASE[-1] / (R_AIR * T)
        rho = rho_base * math.exp(-(H - _H_BASE[-1]) * G0 / (R_AIR * T))
        P = rho * R_AIR * T
    else:
        i = int(np.searchsorted(_H_BASE, H, side="right")) - 1
        i = min(max(i, 0), len(_LAPSE) - 1)
        dH, L = H - _H_BASE[i], _LAPSE[i]
        Tb, Pb = _T_BASE[i], _P_BASE[i]
        T = Tb + L * dH
        if L == 0.0:
            P = Pb * math.exp(-G0 * dH / (R_AIR * Tb))
        else:
            P = Pb * (T / Tb) ** (-G0 / (R_AIR * L))
        rho = P / (R_AIR * T)

    a = math.sqrt(GAMMA_AIR * R_AIR * T)
    return T, P, rho, a


def gravity_eci(r_i: np.ndarray) -> np.ndarray:
    """
    Central inverse-square gravity, exactly the spec form
    g(h) = g0 (Re/(Re+h))^2 directed toward the geocenter:
    g_vec = -g0 Re^2 / |r|^2 * r_hat.
    """
    rn = math.sqrt(r_i[0] ** 2 + r_i[1] ** 2 + r_i[2] ** 2)
    return (-G0 * R_EARTH * R_EARTH / rn ** 3) * r_i

# ==============================================================================
# SECTION 3 -- VEHICLE DEFINITION AND VARIABLE MASS PROPERTIES
# ==============================================================================

@dataclass
class Vehicle:
    """
    Single-stage TVC launch vehicle. All longitudinal stations are measured
    from the NOSE TIP, positive aft [m].

    Propellant model: the tank drains bottom-fixed -- under axial acceleration
    the liquid settles aft, so the remaining column occupies
    [x_tank_bot - l_p(t), x_tank_bot] with l_p proportional to remaining mass.
    This produces the continuous CG travel and inertia decay of the spec.
    """
    # --- mass & propulsion (spec stage) ------------------------------------
    m_wet:     float = 50_000.0     # lift-off mass                     [kg]
    m_dry:     float = 5_000.0      # burnout structural mass           [kg]
    isp:       float = 300.0        # sea-level specific impulse        [s]
    thrust_sl: float = 750e3        # sea-level thrust                  [N]
    t_burn:    float = 120.0        # commanded burn duration           [s]
    a_exit:    float = 1.0          # nozzle exit area                  [m^2]

    # --- geometry ----------------------------------------------------------
    length:     float = 30.0        # overall length                    [m]
    diameter:   float = 3.0         # body diameter                     [m]
    x_cg_dry:   float = 16.0        # dry-structure CG station          [m]
    x_tank_top: float = 17.0        # propellant tank forward station   [m]
    x_tank_bot: float = 24.1        # propellant tank aft station       [m]
    r_tank:     float = 1.42        # tank internal radius              [m]
    x_nozzle:   float = 30.0        # gimbal pivot station              [m]

    # --- aerodynamics ------------------------------------------------------
    c_n_alpha:  float = 4.5         # normal-force slope dCN/da         [1/rad]
    cd_cross:   float = 1.2         # crossflow drag increment coeff    [-]
    c_lp:       float = -1.0        # roll damping derivative           [-]
    c_mq:       float = -35.0       # pitch damping derivative          [-]
    c_nr:       float = -35.0       # yaw damping derivative            [-]

    # Mach tables (piecewise-linear, np.interp clamps at the ends).
    mach_tab: tuple = (0.0, 0.6, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2,
                       1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
    cd_tab:   tuple = (0.28, 0.27, 0.28, 0.32, 0.40, 0.58, 0.60, 0.57,
                       0.48, 0.40, 0.32, 0.29, 0.26, 0.25)
    mach_cp_tab: tuple = (0.0, 0.8, 0.95, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0)
    xcp_tab:     tuple = (20.3, 20.3, 20.8, 21.3, 21.1, 20.5, 19.8, 19.3, 19.0)

    # --- TVC actuator ------------------------------------------------------
    delta_max:      float = math.radians(7.0)    # gimbal hard stop     [rad]
    delta_rate_max: float = math.radians(15.0)   # actuator slew rate   [rad/s]
    tau_tvc:        float = 0.08                 # actuator lag         [s]

    def __post_init__(self) -> None:
        self.s_ref     = math.pi * self.diameter ** 2 / 4.0   # reference area
        self.m_prop0   = self.m_wet - self.m_dry
        self.mdot      = self.thrust_sl / (G0 * self.isp)     # 254.93 kg/s
        self.thrust_vac = self.thrust_sl + P0_SL * self.a_exit
        self.l_tank    = self.x_tank_bot - self.x_tank_top

        # The spec's stage numbers are honored exactly as given: at 750 kN /
        # Isp 300 s the engine consumes mdot*t_burn ~ 30.6 t of the 45 t load,
        # i.e. the engine cuts off on the 120 s command with residual
        # propellant on board (a perfectly consistent flight profile).
        if self.mdot * self.t_burn >= self.m_prop0:
            raise ValueError("Propellant would deplete before commanded cutoff")
        # Sanity: tank must physically hold the propellant at ~LOX/RP-1 bulk
        # density (~1000 kg/m^3).
        tank_vol = math.pi * self.r_tank ** 2 * self.l_tank
        assert 800.0 < self.m_prop0 / tank_vol < 1200.0, "unphysical tank sizing"

    # ------------------------------------------------------------------
    def mass_properties(self, m_prop: float) -> tuple[float, float, np.ndarray]:
        """
        Composite mass, CG station and principal inertia diagonal
        [Ixx, Iyy, Izz] about the instantaneous CG for a given remaining
        propellant mass.

        Dry structure : slender-rod transverse inertia (m L^2 / 12) plus a
                        thin-shell axial term (m r^2), stationed at x_cg_dry.
        Propellant    : solid cylinder, radius r_tank, length l_p(t), aft face
                        fixed at x_tank_bot;  I_trans = m(3r^2 + l^2)/12.
        Both terms are transferred to the composite CG by parallel axis.
        """
        m_prop = max(m_prop, 0.0)
        m = self.m_dry + m_prop

        l_p = self.l_tank * m_prop / self.m_prop0          # column length
        x_p = self.x_tank_bot - 0.5 * l_p                  # column centroid
        x_cg = (self.m_dry * self.x_cg_dry + m_prop * x_p) / m

        r_b = 0.5 * self.diameter
        # Axial (roll) inertia: shell structure + solid liquid column.
        i_xx = self.m_dry * r_b ** 2 + 0.5 * m_prop * self.r_tank ** 2
        # Transverse inertia about the composite CG.
        i_dry = (self.m_dry * self.length ** 2 / 12.0
                 + self.m_dry * (x_cg - self.x_cg_dry) ** 2)
        i_prp = (m_prop * (3.0 * self.r_tank ** 2 + l_p ** 2) / 12.0
                 + m_prop * (x_cg - x_p) ** 2)
        i_tt = i_dry + i_prp
        return m, x_cg, np.array([i_xx, i_tt, i_tt])

# ==============================================================================
# SECTION 4 -- SIMULATION PARAMETERS, LAUNCH FRAME, GUIDANCE
# ==============================================================================

@dataclass
class SimParams:
    """Mission, guidance, control, environment and dispersion parameters."""
    # --- launch site & direction ------------------------------------------
    lat0:    float = math.radians(28.5)      # pad geocentric latitude  [rad]
    lon0:    float = math.radians(-80.55)    # pad longitude            [rad]
    azimuth: float = math.radians(90.0)      # flight azimuth (due east)[rad]

    # --- guidance program --------------------------------------------------
    # A gravity turn integrates the kick over the whole burn, so small kick
    # angles produce large final flight-path rotations (this vehicle's long,
    # low-T/W burn amplifies the kick ~10x by burnout).
    t_kick:       float = 12.0                   # pitchover start      [s]
    theta_kick:   float = math.radians(3.0)      # pitchover kick angle [rad]
    t_pitch_ramp: float = 8.0                    # kick ramp duration   [s]
    v_grav_turn:  float = 30.0                   # min speed for v-track[m/s]

    # --- PD attitude controller  (wn = 2.5 rad/s, zeta = 0.9) -------------
    kp: float = 6.25       # proportional gain = wn^2            [1/s^2]
    kd: float = 4.50       # derivative gain  = 2 zeta wn        [1/s]

    # --- environment -------------------------------------------------------
    wind_ned:  tuple = (0.0, 6.0, 0.0)   # air velocity at 10 m AGL, NED [m/s]
    wind_alpha: float = 0.14             # power-law shear exponent
    wind_maxfac: float = 3.0             # shear amplification cap

    # --- stochastic dispersions (Monte Carlo overrides) --------------------
    thrust_scale: float = 1.0    # thrust & mdot multiplier   (+-2 % MC)
    mass_scale:   float = 1.0    # wet-mass multiplier        (+-1 % MC)
    rho_scale:    float = 1.0    # air-density multiplier     (+-5 % MC)
    d_elev:       float = 0.0    # launch elevation offset    (+-0.5 deg MC)
    d_azim:       float = 0.0    # launch azimuth offset      (+-0.5 deg MC)

    # --- integrator & termination -----------------------------------------
    t_max:   float = 2000.0      # absolute simulation cap    [s]
    q_limit: float = Q_LIMIT_PA  # optimizer max-Q constraint [Pa]


@dataclass
class LaunchFrame:
    """Pad-anchored geometry precomputed once per simulation."""
    r_pad: np.ndarray      # pad position, ECI [m]
    r_iN:  np.ndarray      # 3x3, columns = [north, east, down] in ECI
    up:    np.ndarray      # geodetic up (nominal)
    up_d:  np.ndarray      # dispersed launch/vertical reference
    ahat:  np.ndarray      # dispersed downrange horizontal unit (pitch plane)
    wind_i: np.ndarray     # 10-m wind vector resolved in ECI [m/s]


def make_launch_frame(prm: SimParams) -> LaunchFrame:
    """Build the pad NED triad and the dispersed guidance references."""
    cl, sl = math.cos(prm.lat0), math.sin(prm.lat0)
    co, so = math.cos(prm.lon0), math.sin(prm.lon0)

    r_hat = np.array([cl * co, cl * so, sl])
    north = np.array([-sl * co, -sl * so, cl])
    east  = np.array([-so, co, 0.0])
    down  = -r_hat
    r_iN  = np.column_stack((north, east, down))

    # Dispersed launch references: the whole guidance frame (vertical
    # reference + pitch plane) tilts with the elevation/azimuth dispersion,
    # so a rail misalignment propagates through the entire gravity turn.
    az = prm.azimuth + prm.d_azim
    a_az = north * math.cos(az) + east * math.sin(az)      # downrange horiz.
    up = r_hat
    up_d = math.cos(prm.d_elev) * up + math.sin(prm.d_elev) * a_az
    up_d /= np.linalg.norm(up_d)
    ahat = a_az - (a_az @ up_d) * up_d                     # re-orthogonalize
    ahat /= np.linalg.norm(ahat)

    wind_i = r_iN @ np.asarray(prm.wind_ned, dtype=float)
    return LaunchFrame(r_pad=(R_EARTH + 2.0) * r_hat, r_iN=r_iN, up=up,
                       up_d=up_d, ahat=ahat, wind_i=wind_i)


def initial_state(veh: Vehicle, prm: SimParams, fr: LaunchFrame) -> np.ndarray:
    """15-element initial condition: at rest on the pad, nose along up_d."""
    x_b = fr.up_d                       # body x  : dispersed vertical
    z_b = fr.ahat                       # body z  : downrange (pitch plane)
    y_b = _cross(z_b, x_b)              # body y  : completes RH triad
    q0 = dcm_to_quat(np.column_stack((x_b, y_b, z_b)))
    y = np.zeros(15)
    y[0:3] = fr.r_pad
    y[6:10] = q0
    return y


def guidance_direction(t: float, v_i: np.ndarray, fr: LaunchFrame,
                       prm: SimParams) -> np.ndarray:
    """
    Commanded body-x direction (unit vector, inertial).

    Phase 1  t < t_kick             : hold the launch vertical.
    Phase 2  kick ramp              : half-cosine blend of a theta_kick tilt
                                      into the downrange pitch plane
                                      (the classic pitchover maneuver).
    Phase 3  gravity turn           : track the velocity vector (zero-alpha
                                      flight; gravity curves the trajectory).

    Being open-loop, this profile inherits the classic gravity-turn traits
    (both verified in this model's Monte Carlo): trajectory shape is very
    sensitive to the effective kick (a 0.5 deg rail-elevation error moves
    apogee ~20 km), and low-altitude wind is amplified by the turn -- the
    vehicle weathercocks downwind, which partially absorbs a launch-azimuth
    dispersion into the wind direction. Real vehicles close this loop with
    trajectory-relative steering; modeling that is out of scope here.
    """
    if t < prm.t_kick:
        return fr.up_d
    s = (t - prm.t_kick) / prm.t_pitch_ramp
    if s < 1.0:
        th = prm.theta_kick * 0.5 * (1.0 - math.cos(math.pi * s))
        return math.cos(th) * fr.up_d + math.sin(th) * fr.ahat
    v = math.sqrt(v_i[0] ** 2 + v_i[1] ** 2 + v_i[2] ** 2)
    if v > prm.v_grav_turn:
        return v_i / v
    th = prm.theta_kick
    return math.cos(th) * fr.up_d + math.sin(th) * fr.ahat

# ==============================================================================
# SECTION 5 -- FORCES, MOMENTS, CONTROL AND THE 6-DOF DERIVATIVE
# ==============================================================================

_SIN_CMD_CAP = 0.999          # arcsin domain guard for the TVC inversion
_V_AERO_MIN  = 1.0            # below this airspeed aero forces are ignored
_FD_DT       = 0.01           # central-difference step for I_dot        [s]
_K_QNORM     = 1.0            # quaternion normalization feedback gain [1/s]


def _tvc_command(c_b: np.ndarray, w: np.ndarray, iyy: float, izz: float,
                 thrust: float, l_n: float, prm: SimParams,
                 veh: Vehicle) -> tuple[float, float]:
    """
    PD attitude controller -> commanded gimbal angles.

    Attitude error: e = x_hat x c_b  (c_b = commanded direction in body axes)
    gives the small-angle rotation vector that carries the nose onto the
    command; its y/z components drive pitch/yaw. Desired control moments
      M_des = I * (Kp e - Kd w)
    invert through the exact gimbal geometry (see _forces_body):
      M_y = -T l_n sin(delta_p)   ->  delta_p = asin(-M_des_y / (T l_n))
      M_z = -T l_n sin(delta_y)   ->  delta_y = asin(-M_des_z / (T l_n))
    then saturate at the mechanical stops.
    """
    e_y = -c_b[2]                     # (x_hat x c_b)_y
    e_z = c_b[1]                      # (x_hat x c_b)_z
    m_des_y = iyy * (prm.kp * e_y - prm.kd * w[1])
    m_des_z = izz * (prm.kp * e_z - prm.kd * w[2])
    tl = max(thrust * l_n, 1.0)
    s_p = max(-_SIN_CMD_CAP, min(_SIN_CMD_CAP, -m_des_y / tl))
    s_y = max(-_SIN_CMD_CAP, min(_SIN_CMD_CAP, -m_des_z / tl))
    d_p = max(-veh.delta_max, min(veh.delta_max, math.asin(s_p)))
    d_y = max(-veh.delta_max, min(veh.delta_max, math.asin(s_y)))
    return d_p, d_y


def _forces_body(t: float, y: np.ndarray, veh: Vehicle, prm: SimParams,
                 fr: LaunchFrame, powered: bool, want_aux: bool = False):
    """
    Assemble body-frame force F_B, moment M_B about the instantaneous CG,
    the actuator rates, and (optionally) a telemetry dictionary.
    """
    r = y[0:3]
    v = y[3:6]
    q = quat_normalize(y[6:10])
    w = y[10:13]
    delta_p, delta_y = y[13], y[14]

    # ---- mass state -------------------------------------------------------
    mdot_eff = veh.mdot * prm.thrust_scale
    m_prop0_eff = veh.m_wet * prm.mass_scale - veh.m_dry
    m_prop = max(m_prop0_eff - mdot_eff * min(t, veh.t_burn), 0.0)
    m, x_cg, inertia = veh.mass_properties(m_prop)

    # ---- environment ------------------------------------------------------
    alt = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2) - R_EARTH
    T_amb, p_amb, rho, a_snd = atmosphere(alt)
    rho *= prm.rho_scale                        # MC density perturbation

    r_ib = quat_to_dcm(q)                       # body -> inertial

    # Altitude-dependent horizontal wind (power-law shear, capped).
    wfac = min((max(alt, 10.0) / 10.0) ** prm.wind_alpha, prm.wind_maxfac)
    v_rel_i = v - wfac * fr.wind_i
    v_rel_b = r_ib.T @ v_rel_i
    v_air = math.sqrt(v_rel_b[0] ** 2 + v_rel_b[1] ** 2 + v_rel_b[2] ** 2)

    mach = v_air / a_snd
    q_dyn = 0.5 * rho * v_air * v_air

    f_b = np.zeros(3)
    m_b = np.zeros(3)
    alpha = beta = alpha_t = 0.0
    x_cp = veh.xcp_tab[0]

    # ---- aerodynamics (total angle-of-attack body-of-revolution model) ----
    if v_air > _V_AERO_MIN and rho > 0.0:
        vhat = v_rel_b / v_air
        alpha   = math.atan2(v_rel_b[2], v_rel_b[0])          # pitch-plane AoA
        beta    = math.atan2(v_rel_b[1],
                             math.hypot(v_rel_b[0], v_rel_b[2]))  # sideslip
        alpha_t = math.acos(max(-1.0, min(1.0, vhat[0])))     # total AoA

        s_at = math.sin(alpha_t)
        cd = float(np.interp(mach, veh.mach_tab, veh.cd_tab)) \
            + veh.cd_cross * s_at * s_at                      # crossflow drag
        f_aero = -q_dyn * veh.s_ref * cd * vhat               # drag, anti-vel.

        # Normal force N = Q S CNa sin(a_t), directed opposite the cross-axis
        # component of the body's velocity through the air. Small angles:
        # CL = CNa*alpha (lift), CY = -CNa*beta (side force) -- same model.
        perp = vhat.copy()
        perp[0] = 0.0
        p_norm = math.sqrt(perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2)
        if p_norm > 1e-9:
            f_aero = f_aero - (q_dyn * veh.s_ref * veh.c_n_alpha * s_at
                               / p_norm) * perp

        # Applied at the Mach-dependent center of pressure.
        x_cp = float(np.interp(mach, veh.mach_cp_tab, veh.xcp_tab))
        r_cp_b = np.array([x_cg - x_cp, 0.0, 0.0])
        m_aero = _cross(r_cp_b, f_aero)

        # Rate damping (Clp, Cmq, Cnr), nondimensionalized by d/(2V).
        dd = veh.diameter
        coef = q_dyn * veh.s_ref * dd * dd / (2.0 * max(v_air, 20.0))
        m_damp = coef * np.array([veh.c_lp * w[0],
                                  veh.c_mq * w[1],
                                  veh.c_nr * w[2]])
        f_b += f_aero
        m_b += m_aero + m_damp

    # ---- propulsion + TVC + variable-mass torques -------------------------
    thrust = 0.0
    d_p_cmd = d_y_cmd = 0.0
    delta_dot = np.zeros(2)
    idot = np.zeros(3)
    l_n = veh.x_nozzle - x_cg

    if powered:
        # Ambient-pressure thrust law; thrust_scale also scales mdot so the
        # dispersed engine keeps its Isp.
        thrust = prm.thrust_scale * (veh.thrust_vac - p_amb * veh.a_exit)

        # Exact gimbal geometry: thrust unit in body axes for deflections
        # delta_p (about +y_B) and delta_y (about +z_B):
        cp_, sp_ = math.cos(delta_p), math.sin(delta_p)
        cy_, sy_ = math.cos(delta_y), math.sin(delta_y)
        t_hat = np.array([cp_ * cy_, sy_, -sp_ * cy_])
        f_thr = thrust * t_hat
        r_noz_b = np.array([-l_n, 0.0, 0.0])
        m_b += _cross(r_noz_b, f_thr)
        f_b += f_thr

        # Jet damping: expelled mass at lever l_n removes transverse angular
        # momentum at rate mdot*l_n^2*w (stabilizing).
        m_b += -mdot_eff * l_n * l_n * np.array([0.0, w[1], w[2]])

        # Inertia rate I_dot (central difference through the mass model);
        # contributes the -I_dot w torque of d(Iw)/dt = M.
        _, _, i_fwd = veh.mass_properties(m_prop - mdot_eff * _FD_DT)
        _, _, i_bwd = veh.mass_properties(m_prop + mdot_eff * _FD_DT)
        idot = (i_fwd - i_bwd) / (2.0 * _FD_DT)

        # Guidance + PD control -> commanded gimbal; actuator integrates a
        # rate-limited first-order lag toward the command.
        c_i = guidance_direction(t, v, fr, prm)
        c_b = r_ib.T @ c_i
        d_p_cmd, d_y_cmd = _tvc_command(c_b, w, inertia[1], inertia[2],
                                        thrust, l_n, prm, veh)
        rate = veh.delta_rate_max
        delta_dot = np.array([
            max(-rate, min(rate, (d_p_cmd - delta_p) / veh.tau_tvc)),
            max(-rate, min(rate, (d_y_cmd - delta_y) / veh.tau_tvc)),
        ])
    else:
        # Coast: a de-energized gimbal has no dynamical effect (thrust = 0),
        # but relax it to null rather than freezing the state so the TVC
        # telemetry does not show a phantom constant deflection.
        rate = veh.delta_rate_max
        delta_dot = np.array([
            max(-rate, min(rate, -delta_p / veh.tau_tvc)),
            max(-rate, min(rate, -delta_y / veh.tau_tvc)),
        ])

    aux = None
    if want_aux:
        aux = dict(m=m, x_cg=x_cg, x_cp=x_cp, alt=alt, v_air=v_air, mach=mach,
                   q_dyn=q_dyn, thrust=thrust, alpha=alpha, beta=beta,
                   alpha_t=alpha_t, rho=rho, p_amb=p_amb,
                   d_p_cmd=d_p_cmd, d_y_cmd=d_y_cmd)
    return f_b, m_b, delta_dot, m, inertia, idot, r_ib, w, aux


def dynamics(t: float, y: np.ndarray, veh: Vehicle, prm: SimParams,
             fr: LaunchFrame, powered: bool) -> np.ndarray:
    """Full 15-state derivative (see module docstring for the expanded EOM)."""
    (f_b, m_b, delta_dot, m, inertia, idot,
     r_ib, w, _) = _forces_body(t, y, veh, prm, fr, powered)

    q = quat_normalize(y[6:10])

    # Translation: m v_dot = R_IB F_B + m g  (thrust already carries the
    # propellant momentum flux; standard variable-mass rocket equation).
    a_i = (r_ib @ f_b) / m + gravity_eci(y[0:3])

    # Attitude kinematics with normalization feedback.
    q_dot = 0.5 * quat_mult(q, np.array([0.0, w[0], w[1], w[2]]))
    q_raw = y[6:10]
    q_dot += _K_QNORM * (1.0 - float(q_raw @ q_raw)) * q_raw

    # Rotation: I w_dot = M - w x (I w) - I_dot w   (diagonal I).
    iw = inertia * w
    gyro = _cross(w, iw)
    w_dot = (m_b - gyro - idot * w) / inertia

    out = np.empty(15)
    out[0:3] = y[3:6]
    out[3:6] = a_i
    out[6:10] = q_dot
    out[10:13] = w_dot
    out[13:15] = delta_dot
    return out

# ==============================================================================
# SECTION 6 -- SIMULATION DRIVER
# ==============================================================================

# (rtol, burn-phase sample dt, coast-phase sample dt) per run mode.
_MODE_SET = {
    "full":   (1e-6, 0.25, 1.0),    # nominal runs: dense telemetry
    "mc":     (1e-5, 0.50, 2.0),    # Monte Carlo: faster, lighter output
    "apogee": (1e-7, 0.25, 1.0),    # optimizer: tight tol, stops at apogee
}

_ATOL = np.array([1e-1] * 3 + [1e-3] * 3 + [1e-8] * 4 + [1e-6] * 3 + [1e-6] * 2)


@dataclass
class SimResult:
    """Time histories plus scalar flight metrics for one trajectory."""
    mode: str
    t: np.ndarray
    y: np.ndarray                    # 15 x N state history
    alt: np.ndarray
    speed: np.ndarray
    ned_km: np.ndarray               # 3 x N pad-relative [N, E, alt] in km
    apogee_alt: float
    t_apogee: float
    max_q: float                     # ASCENT max-Q (structural constraint)
    t_max_q: float
    reentry_max_q: float             # peak Q of the ballistic return (info)
    t_impact: float | None
    impact_en_km: tuple | None       # (east, north) at impact, km
    downrange_km: float | None
    burnout: dict
    prm: SimParams
    # dense telemetry (mode == 'full' only)
    mach: np.ndarray | None = None
    q_dyn: np.ndarray | None = None
    thrust: np.ndarray | None = None
    mass: np.ndarray | None = None
    x_cg: np.ndarray | None = None
    x_cp: np.ndarray | None = None
    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None
    euler_deg: np.ndarray | None = None
    fpa_deg: np.ndarray | None = None


def _refine_peak(t: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """Parabolic refinement of a sampled maximum (keeps the optimizer's
    max-Q constraint smooth in the design variables)."""
    i = int(np.argmax(f))
    if 0 < i < len(f) - 1:
        t0, t1, t2 = t[i - 1], t[i], t[i + 1]
        f0, f1, f2 = f[i - 1], f[i], f[i + 1]
        denom = (f0 - 2.0 * f1 + f2)
        if abs(denom) > 1e-12:
            s = 0.5 * (f0 - f2) / denom
            s = max(-1.0, min(1.0, s))
            dt = t1 - t0
            return t1 + s * dt, f1 - 0.25 * (f0 - f2) * s
    return float(t[i]), float(f[i])


def simulate(veh: Vehicle, prm: SimParams, mode: str = "full") -> SimResult:
    """
    Integrate one complete flight with scipy's adaptive RK45.

    Phase 1 (powered)  : t in [0, t_burn]; thrust + TVC active.
    Phase 2 (coast)    : t in [t_burn, t_max]; terminated by the ground-impact
                         event (or the apogee event in 'apogee' mode).
    The phase split places the thrust-cutoff discontinuity exactly on a mesh
    boundary, which the adaptive integrator requires for clean error control.
    """
    rtol, dt_burn, dt_coast = _MODE_SET[mode]
    fr = make_launch_frame(prm)
    y0 = initial_state(veh, prm, fr)
    tb = veh.t_burn

    # ---- phase 1: powered ascent -----------------------------------------
    sol1 = solve_ivp(dynamics, (0.0, tb), y0, method="RK45",
                     t_eval=np.arange(0.0, tb + 1e-9, dt_burn),
                     rtol=rtol, atol=_ATOL, max_step=2.0,
                     args=(veh, prm, fr, True))
    if not sol1.success:
        raise RuntimeError(f"Powered-phase integration failed: {sol1.message}")

    # ---- phase 2: coast / ballistic --------------------------------------
    def ev_impact(t, y, *_):
        return math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2) - R_EARTH
    ev_impact.terminal = True
    ev_impact.direction = -1

    def ev_apogee(t, y, *_):
        r = y[0:3]
        rn = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        return float(y[3:6] @ r) / rn
    ev_apogee.terminal = (mode == "apogee")
    ev_apogee.direction = -1

    sol2 = solve_ivp(dynamics, (tb, prm.t_max), sol1.y[:, -1], method="RK45",
                     t_eval=np.arange(tb, prm.t_max, dt_coast),
                     events=(ev_impact, ev_apogee),
                     rtol=rtol, atol=_ATOL, max_step=10.0,
                     args=(veh, prm, fr, False))
    if not sol2.success and sol2.status != 1:
        raise RuntimeError(f"Coast-phase integration failed: {sol2.message}")

    # ---- stitch phases (drop duplicated t_burn sample) --------------------
    t = np.concatenate([sol1.t, sol2.t[1:] if len(sol2.t) else []])
    ys = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1) \
        if sol2.y.shape[1] else sol1.y

    # Append exact terminal-event states (impact, or apogee in apogee mode).
    t_impact = None
    if len(sol2.t_events[0]):
        t_ev = float(sol2.t_events[0][0])
        t = np.append(t, t_ev)
        ys = np.column_stack([ys, sol2.y_events[0][0]])
        t_impact = t_ev
    t_apogee_ev = None
    if len(sol2.t_events[1]):
        t_apogee_ev = float(sol2.t_events[1][0])
        if mode == "apogee":
            t = np.append(t, t_apogee_ev)
            ys = np.column_stack([ys, sol2.y_events[1][0]])

    # ---- basic kinematic telemetry ---------------------------------------
    n = len(t)
    alt = np.linalg.norm(ys[0:3, :], axis=0) - R_EARTH
    speed = np.linalg.norm(ys[3:6, :], axis=0)

    # Pad-relative NED projection (E/N horizontal, true altitude vertical).
    rel = ys[0:3, :] - fr.r_pad[:, None]
    ned = fr.r_iN.T @ rel
    ned_km = np.vstack([ned[0] / 1e3, ned[1] / 1e3, alt / 1e3])

    # Dynamic pressure needs rho(h) and airspeed at every sample.
    q_dyn = np.empty(n)
    mach_arr = np.empty(n)
    for k in range(n):
        _, _, rho_k, a_k = atmosphere(alt[k])
        rho_k *= prm.rho_scale
        wfac = min((max(alt[k], 10.0) / 10.0) ** prm.wind_alpha,
                   prm.wind_maxfac)
        v_rel = ys[3:6, k] - wfac * fr.wind_i
        v_air = float(np.linalg.norm(v_rel))
        q_dyn[k] = 0.5 * rho_k * v_air * v_air
        mach_arr[k] = v_air / a_k

    # Apogee: exact event when recorded, else best sample.
    if t_apogee_ev is not None:
        t_apo = t_apogee_ev
        apo = float(np.interp(t_apo, t, alt)) if mode != "apogee" else float(alt[-1])
    else:
        i_apo = int(np.argmax(alt))
        t_apo, apo = float(t[i_apo]), float(alt[i_apo])

    # Max-Q is an ASCENT (boost-phase) structural metric; the ballistic
    # return re-enters at high Mach and its far larger Q peak is reported
    # separately so it cannot masquerade as the design constraint.
    asc = t <= t_apo + 1e-9
    t_max_q, max_q = _refine_peak(t[asc], q_dyn[asc])
    reentry_max_q = float(q_dyn[~asc].max()) if np.any(~asc) else 0.0

    # Impact location in the pad frame + great-circle downrange.
    impact_en = None
    downrange = None
    if t_impact is not None:
        r_end = ys[0:3, -1]
        p_ned = fr.r_iN.T @ (r_end - fr.r_pad)
        impact_en = (float(p_ned[1] / 1e3), float(p_ned[0] / 1e3))
        # atan2 form of the great-circle angle: numerically robust at small
        # separations (acos loses precision near cos ~ 1).
        u1 = r_end / np.linalg.norm(r_end)
        u0 = fr.r_pad / np.linalg.norm(fr.r_pad)
        downrange = R_EARTH * math.atan2(np.linalg.norm(_cross(u0, u1)),
                                         float(u0 @ u1)) / 1e3

    # Burnout snapshot (last powered sample).
    i_bo = int(np.searchsorted(t, tb))
    i_bo = min(i_bo, n - 1)
    burnout = dict(t=float(t[i_bo]), alt=float(alt[i_bo]),
                   speed=float(speed[i_bo]), mach=float(mach_arr[i_bo]))

    res = SimResult(mode=mode, t=t, y=ys, alt=alt, speed=speed, ned_km=ned_km,
                    apogee_alt=apo, t_apogee=t_apo, max_q=float(max_q),
                    t_max_q=float(t_max_q), reentry_max_q=reentry_max_q,
                    t_impact=t_impact,
                    impact_en_km=impact_en, downrange_km=downrange,
                    burnout=burnout, prm=prm, mach=mach_arr, q_dyn=q_dyn)

    # ---- dense telemetry (nominal/optimized runs only) --------------------
    if mode == "full":
        thrust = np.empty(n); mass = np.empty(n)
        xcg = np.empty(n); xcp = np.empty(n)
        al = np.empty(n); be = np.empty(n)
        euler = np.empty((3, n)); fpa = np.empty(n)
        for k in range(n):
            powered = t[k] < tb
            aux = _forces_body(t[k], ys[:, k], veh, prm, fr, powered,
                               want_aux=True)[-1]
            thrust[k], mass[k] = aux["thrust"], aux["m"]
            xcg[k], xcp[k] = aux["x_cg"], aux["x_cp"]
            al[k], be[k] = aux["alpha"], aux["beta"]
            r_nb = fr.r_iN.T @ quat_to_dcm(quat_normalize(ys[6:10, k]))
            roll, pitch, yaw = euler_zyx_from_dcm(r_nb)
            euler[:, k] = np.degrees([roll, pitch, yaw])
            vk = speed[k]
            fpa[k] = math.degrees(math.asin(
                float(ys[3:6, k] @ ys[0:3, k])
                / max(vk * np.linalg.norm(ys[0:3, k]), 1e-9))) if vk > 1 else 90.0
        res.thrust, res.mass = thrust, mass
        res.x_cg, res.x_cp = xcg, xcp
        res.alpha, res.beta = np.degrees(al), np.degrees(be)
        res.euler_deg, res.fpa_deg = euler, fpa
    return res

# ==============================================================================
# SECTION 7 -- MONTE CARLO UNCERTAINTY ANALYSIS ENGINE
# ==============================================================================

def run_monte_carlo(veh: Vehicle, prm: SimParams,
                    n_runs: int = N_MONTE_CARLO) -> dict:
    """
    N stochastic runs with independent uniform dispersions per the spec:
      thrust +-2 %, lift-off mass +-1 %, air density +-5 %,
      launch elevation and azimuth +-0.5 deg.
    Returns per-run scalar metrics plus decimated 3-D trajectories.
    """
    rng = np.random.default_rng(RNG_SEED)
    out = dict(apogee=np.empty(n_runs), max_q=np.empty(n_runs),
               downrange=np.empty(n_runs), impact_e=np.empty(n_runs),
               impact_n=np.empty(n_runs), trajs=[])
    t0 = time.perf_counter()
    for i in range(n_runs):
        prm_i = replace(
            prm,
            thrust_scale=1.0 + rng.uniform(-0.02, 0.02),
            mass_scale=1.0 + rng.uniform(-0.01, 0.01),
            rho_scale=1.0 + rng.uniform(-0.05, 0.05),
            d_elev=rng.uniform(-1.0, 1.0) * math.radians(0.5),
            d_azim=rng.uniform(-1.0, 1.0) * math.radians(0.5),
        )
        res = simulate(veh, prm_i, mode="mc")
        out["apogee"][i] = res.apogee_alt
        out["max_q"][i] = res.max_q
        out["downrange"][i] = res.downrange_km if res.downrange_km else np.nan
        if res.impact_en_km:
            out["impact_e"][i], out["impact_n"][i] = res.impact_en_km
        else:                                   # no impact before t_max
            out["impact_e"][i] = out["impact_n"][i] = np.nan
        out["trajs"].append(res.ned_km[:, ::4])
        if (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"    MC progress: {i + 1:3d}/{n_runs}   "
                  f"({el:5.1f} s elapsed)")
    return out

# ==============================================================================
# SECTION 8 -- TRAJECTORY OPTIMIZATION (PITCHOVER PROGRAM)
# ==============================================================================

# Design-variable scaling: x = [t_kick / T_SCALE, theta_kick / TH_SCALE].
_T_SCALE, _TH_SCALE = 40.0, 0.6


def optimize_pitchover(veh: Vehicle, prm: SimParams) -> dict:
    """
    scipy.optimize.minimize (SLSQP) over the pitchover program (t_kick,
    theta_kick):

        maximize   insertion velocity  (inertial speed at burnout)
        subject to ascent max-Q <= Q_LIMIT_PA
        bounds     t_kick in [3, 40] s,  theta_kick in [1, 35] deg.

    Of the two spec objectives, insertion velocity is the one this constraint
    genuinely fights: pitching over earlier/harder cuts gravity losses (more
    burnout speed) but holds the vehicle faster in denser air (more max-Q),
    so the optimum rides the max-Q boundary. (Apogee maximization is
    degenerate for a gravity-turn: apogee and max-Q are both monotone toward
    the vertical, and the "optimum" just pins the bounds.)

    Each function evaluation is a full 6-DOF flight, terminated at the
    apogee event for economy; runs are memoized so the objective and the
    constraint share simulations.
    """
    cache: dict = {}

    def run(x) -> SimResult:
        key = (float(x[0]), float(x[1]))
        if key not in cache:
            p = replace(prm, t_kick=float(x[0]) * _T_SCALE,
                        theta_kick=float(x[1]) * _TH_SCALE)
            cache[key] = simulate(veh, p, mode="apogee")
        return cache[key]

    def objective(x) -> float:
        return -run(x).burnout["speed"] / 1e3    # O(1) scaling for SLSQP

    def q_constraint(x) -> float:
        return (prm.q_limit - run(x).max_q) / 1e4   # >= 0 feasible

    x0 = np.array([prm.t_kick / _T_SCALE, prm.theta_kick / _TH_SCALE])
    bounds = [(3.0 / _T_SCALE, 1.0),
              (math.radians(1.0) / _TH_SCALE, math.radians(35.0) / _TH_SCALE)]

    t0 = time.perf_counter()
    sol = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": q_constraint}],
                   options={"eps": 4e-3, "ftol": 1e-4, "maxiter": 40})
    elapsed = time.perf_counter() - t0

    best = run(sol.x)
    return dict(t_kick=float(sol.x[0]) * _T_SCALE,
                theta_kick_deg=math.degrees(float(sol.x[1]) * _TH_SCALE),
                v_burnout=best.burnout["speed"],
                apogee_km=best.apogee_alt / 1e3,
                max_q_kpa=best.max_q / 1e3,
                n_sims=len(cache), success=bool(sol.success),
                message=str(sol.message), elapsed=elapsed,
                nfev=int(sol.nfev))

# ==============================================================================
# SECTION 9 -- VISUALIZATION SUITE
# ==============================================================================

def _apply_style() -> None:
    """House chart style: recessive grid/axes, ink-on-surface text."""
    plt.rcParams.update({
        "figure.facecolor": COL["surface"], "axes.facecolor": COL["surface"],
        "savefig.facecolor": COL["surface"],
        "axes.edgecolor": COL["axis"], "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": COL["ink"], "axes.labelcolor": COL["sub"],
        "axes.titlecolor": COL["ink"],
        "xtick.color": COL["muted"], "ytick.color": COL["muted"],
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "lines.linewidth": 1.6,
    })


def _style_3d(ax) -> None:
    """Recessive panes and grid for 3-D axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=COL["grid"], linewidth=0.6)
    ax.tick_params(colors=COL["muted"], labelsize=7)


def _guard_crossrange_axis(ax, e_vals: np.ndarray, n_vals: np.ndarray) -> None:
    """
    A nominal trajectory is planar to machine precision, so autoscale would
    stretch ~1e-12 km of numerical noise across the whole North axis and
    render the flight path as a jagged ribbon. Enforce a sane minimum
    crossrange extent (fraction of the downrange span).
    """
    e_span = float(np.ptp(e_vals))
    n_lo, n_hi = float(np.min(n_vals)), float(np.max(n_vals))
    width = max(n_hi - n_lo, 0.10 * e_span, 4.0)
    center = 0.5 * (n_lo + n_hi)
    ax.set_ylim(center - 0.55 * width, center + 0.55 * width)


def plot_trajectory_3d(nom: SimResult, opt: SimResult | None,
                       fname: str) -> None:
    """3-D flight path in the pad frame, nominal vs. optimized."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    _style_3d(ax)

    for res, color, label in ((nom, COL["s1"], "Nominal"),
                              (opt, COL["s2"], "Optimized")):
        if res is None:
            continue
        n_, e_, u_ = res.ned_km
        ax.plot(e_, n_, u_, color=color, lw=1.8, label=label)
        ax.plot(e_, n_, np.zeros_like(u_), color=color, lw=0.9,
                ls="--", alpha=0.35)                       # ground track
        # Flight milestones, directly labeled.
        for t_m, mk, name in ((res.t_max_q, "D", "max-Q"),
                              (res.burnout["t"], "s", "burnout"),
                              (res.t_apogee, "o", "apogee")):
            ii = int(np.argmin(np.abs(res.t - t_m)))
            ax.scatter(e_[ii], n_[ii], u_[ii], marker=mk, s=28, color=color,
                       depthshade=False)
            if res is nom:
                ax.text(e_[ii], n_[ii], u_[ii] + 6, name, fontsize=7,
                        color=COL["sub"])
        if res.impact_en_km:
            ax.scatter(*res.impact_en_km, 0, marker="x", s=40, color=COL["crit"],
                       depthshade=False)

    e_all = np.concatenate([r.ned_km[1] for r in (nom, opt) if r is not None])
    n_all = np.concatenate([r.ned_km[0] for r in (nom, opt) if r is not None])
    _guard_crossrange_axis(ax, e_all, n_all)
    ax.set_xlabel("East [km]"); ax.set_ylabel("North [km]")
    ax.set_zlabel("Altitude [km]")
    ax.set_title("6-DOF ascent trajectory -- pad-relative frame")
    ax.legend(loc="upper left")
    ax.set_box_aspect((1.4, 1.0, 0.7))
    fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
    print(f"    saved {fname}")


def plot_dashboard(res: SimResult, fname: str) -> None:
    """Nine-panel telemetry dashboard for one densely sampled flight."""
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    t = res.t
    tb = res.burnout["t"]

    def vline(ax):
        ax.axvline(tb, color=COL["muted"], lw=0.8, ls=":", zorder=1)

    p = axs[0, 0]; p.plot(t, res.alt / 1e3, color=COL["s1"]); vline(p)
    p.set_title("Altitude [km]")
    p.annotate(f"apogee {res.apogee_alt / 1e3:.1f} km",
               xy=(res.t_apogee, res.apogee_alt / 1e3),
               xytext=(-10, -12), textcoords="offset points",
               fontsize=7, color=COL["sub"], ha="right")

    p = axs[0, 1]; p.plot(t, res.mach, color=COL["s1"]); vline(p)
    p.set_title("Mach number [-]")

    p = axs[0, 2]; p.plot(t, res.q_dyn / 1e3, color=COL["s1"]); vline(p)
    p.axhline(Q_LIMIT_PA / 1e3, color=COL["crit"], lw=1.0, ls="--")
    p.annotate("optimizer max-Q limit", xy=(0.98, Q_LIMIT_PA / 1e3),
               xycoords=("axes fraction", "data"), xytext=(0, 4),
               textcoords="offset points", fontsize=7, color=COL["crit"],
               ha="right")
    p.annotate(f"ascent max-Q {res.max_q / 1e3:.1f} kPa @ {res.t_max_q:.0f} s",
               xy=(res.t_max_q, res.max_q / 1e3), xytext=(8, 2),
               textcoords="offset points", fontsize=7, color=COL["sub"])
    # Clip the axis to the ascent scale: the ballistic-return spike is real
    # but two orders larger, and would flatten the boost-phase detail.
    y_top = 1.25 * max(res.max_q, Q_LIMIT_PA) / 1e3
    if res.reentry_max_q > y_top * 1e3:
        p.set_ylim(0.0, y_top)
        p.annotate(f"reentry peak {res.reentry_max_q / 1e3:.0f} kPa (off scale)",
                   xy=(0.98, 0.02), xycoords="axes fraction", fontsize=7,
                   color=COL["muted"], ha="right")
    p.set_title("Dynamic pressure Q [kPa]")

    p = axs[1, 0]; p.plot(t, res.thrust / 1e3, color=COL["s1"]); vline(p)
    p.set_title("Thrust [kN]")

    p = axs[1, 1]; p.plot(t, res.mass / 1e3, color=COL["s1"]); vline(p)
    p.set_title("Total mass [t]")

    p = axs[1, 2]
    for i, (nm, c) in enumerate((("roll", COL["s1"]), ("pitch", COL["s2"]),
                                 ("yaw", COL["s3"]))):
        p.plot(t, res.euler_deg[i], color=c, label=nm)
    vline(p); p.legend(loc="best"); p.set_title("Euler angles (NED) [deg]")

    p = axs[2, 0]
    p.plot(t, np.degrees(res.y[13]), color=COL["s1"], label=r"$\delta_p$")
    p.plot(t, np.degrees(res.y[14]), color=COL["s2"], label=r"$\delta_y$")
    for s in (+1, -1):
        p.axhline(s * math.degrees(Vehicle.delta_max), color=COL["crit"],
                  lw=0.8, ls="--", alpha=0.7)
    vline(p); p.legend(loc="best"); p.set_title("TVC gimbal deflection [deg]")

    p = axs[2, 1]
    for i, (nm, c) in enumerate((("p", COL["s1"]), ("q", COL["s2"]),
                                 ("r", COL["s3"]))):
        p.plot(t, np.degrees(res.y[10 + i]), color=c, label=nm)
    vline(p); p.legend(loc="best"); p.set_title("Body rates [deg/s]")

    p = axs[2, 2]
    p.plot(t, res.alpha, color=COL["s1"], label=r"$\alpha$")
    p.plot(t, res.beta, color=COL["s2"], label=r"$\beta$")
    vline(p); p.legend(loc="best")
    p.set_title("Incidence angles [deg]")

    for ax in axs.flat:
        ax.set_xlabel("time [s]")
    fig.suptitle("Nominal mission telemetry (dotted line = burnout)",
                 fontsize=12)
    fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
    print(f"    saved {fname}")


def _dispersion_ellipses(ax, e: np.ndarray, n: np.ndarray) -> None:
    """1/2/3-sigma covariance ellipses of the impact scatter."""
    pts = np.vstack([e, n])
    mu = pts.mean(axis=1)
    cov = np.cov(pts)
    evals, evecs = np.linalg.eigh(cov)
    ang = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1]))
    for k, a_ in ((1, 0.9), (2, 0.55), (3, 0.3)):
        ax.add_patch(Ellipse(mu, 2 * k * math.sqrt(evals[1]),
                             2 * k * math.sqrt(evals[0]), angle=ang,
                             fill=False, color=COL["s1"], lw=1.1, alpha=a_))


def plot_monte_carlo(mc: dict, nom: SimResult, fname: str) -> None:
    """Dispersion suite: 3-D spaghetti, impact ellipse, apogee statistics."""
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    _style_3d(ax)
    for tr in mc["trajs"]:
        ax.plot(tr[1], tr[0], tr[2], color=COL["muted"], lw=0.6, alpha=0.30)
    n_, e_, u_ = nom.ned_km
    ax.plot(e_, n_, u_, color=COL["s1"], lw=1.8)
    _guard_crossrange_axis(ax,
                           np.concatenate([tr[1] for tr in mc["trajs"]]),
                           np.concatenate([tr[0] for tr in mc["trajs"]]))
    ax.set_xlabel("East [km]"); ax.set_ylabel("North [km]")
    ax.set_zlabel("Alt [km]")
    ax.set_title(f"Trajectory dispersion (N = {len(mc['trajs'])})")
    ax.set_box_aspect((1.4, 1.0, 0.7))

    ax = fig.add_subplot(2, 2, 2)
    ok = ~np.isnan(mc["impact_e"])
    ax.scatter(mc["impact_e"][ok], mc["impact_n"][ok], s=14,
               color=COL["muted"], alpha=0.75, edgecolors="none")
    if nom.impact_en_km:
        ax.scatter(*nom.impact_en_km, marker="*", s=140, color=COL["s1"],
                   zorder=5, label="nominal")
        ax.legend(loc="best")
    _dispersion_ellipses(ax, mc["impact_e"][ok], mc["impact_n"][ok])
    ax.set_xlabel("East [km]"); ax.set_ylabel("North [km]")
    ax.set_title("Impact dispersion with 1/2/3-sigma ellipses")
    # Downrange scatter dwarfs crossrange ~70:1 (open-loop gravity turn
    # weathercocks into the wind plane); an equal-aspect view would collapse
    # to a line, so the axes are deliberately independent.
    ax.annotate("note: axis scales differ", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=7, color=COL["muted"])

    ax = fig.add_subplot(2, 2, 3)
    ax.hist(mc["apogee"] / 1e3, bins=16, color=COL["s1"],
            edgecolor=COL["surface"], linewidth=0.8)
    ax.axvline(nom.apogee_alt / 1e3, color=COL["s2"], lw=1.4)
    ax.annotate("nominal", xy=(nom.apogee_alt / 1e3, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -10),
                textcoords="offset points", fontsize=7, color=COL["s2"])
    ax.set_xlabel("Apogee altitude [km]"); ax.set_ylabel("runs")
    ax.set_title("Apogee distribution")

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(mc["downrange"][ok], mc["apogee"][ok] / 1e3, s=14,
               color=COL["muted"], alpha=0.75, edgecolors="none")
    if nom.downrange_km:
        ax.scatter(nom.downrange_km, nom.apogee_alt / 1e3, marker="*", s=140,
                   color=COL["s1"], zorder=5)
    ax.set_xlabel("Impact downrange [km]"); ax.set_ylabel("Apogee [km]")
    ax.set_title("Apogee vs. downrange")

    fig.suptitle("Monte Carlo uncertainty analysis "
                 "(thrust +-2 %, mass +-1 %, rho +-5 %, rail +-0.5 deg)",
                 fontsize=12)
    fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
    print(f"    saved {fname}")

# ==============================================================================
# SECTION 10 -- MODEL SELF-VERIFICATION
# ==============================================================================

def _self_test() -> None:
    """
    Numerical verification of the physics kernels against published values
    and mathematical identities. Raises AssertionError on any failure.
    """
    # --- USSA-76 vs. published table (geometric h from geopotential H) -----
    def h_geom(H):
        return R0_GEOPOT * H / (R0_GEOPOT - H)
    for H, T_ref, P_ref in ((11_000.0, 216.65, 22_632.06),
                            (20_000.0, 216.65, 5_474.889),
                            (32_000.0, 228.65, 868.0187),
                            (47_000.0, 270.65, 110.9063)):
        T, P, _, _ = atmosphere(h_geom(H))
        assert abs(T - T_ref) < 0.02, f"USSA T at H={H}: {T} != {T_ref}"
        assert abs(P / P_ref - 1.0) < 3e-3, f"USSA P at H={H}: {P} != {P_ref}"
    T0, P0, rho0, a0 = atmosphere(0.0)
    assert abs(rho0 - 1.2250) < 1e-3 and abs(a0 - 340.29) < 0.05

    # --- quaternion algebra ------------------------------------------------
    rng = np.random.default_rng(7)
    for _ in range(50):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        th = rng.uniform(-math.pi, math.pi)
        q = np.array([math.cos(th / 2), *(math.sin(th / 2) * ax)])
        R = quat_to_dcm(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)      # orthonormal
        assert abs(np.linalg.det(R) - 1.0) < 1e-12              # proper
        q2 = dcm_to_quat(R)
        assert abs(abs(float(q @ q2)) - 1.0) < 1e-9             # round trip

    # --- mass properties & propulsion consistency --------------------------
    veh = Vehicle()
    m0, xcg0, i0 = veh.mass_properties(veh.m_prop0)
    assert abs(m0 - veh.m_wet) < 1e-9
    assert 0.0 < xcg0 < veh.length and np.all(i0 > 0)
    assert abs(veh.mdot * G0 * veh.isp - veh.thrust_sl) < 1e-6
    m_bo = veh.m_prop0 - veh.mdot * veh.t_burn
    assert m_bo > 0.0                       # residual-propellant cutoff
    # CG must remain between the dry CG and the tank aft face.
    _, xcg_bo, _ = veh.mass_properties(m_bo)
    assert veh.x_cg_dry < xcg_bo < veh.x_tank_bot

    # --- gravity -----------------------------------------------------------
    g_surface = np.linalg.norm(gravity_eci(np.array([R_EARTH, 0.0, 0.0])))
    assert abs(g_surface - G0) < 1e-9
    print("    all physics self-tests passed "
          "(USSA-76 tables, quaternion algebra, mass/propulsion, gravity)")

# ==============================================================================
# SECTION 11 -- MAIN ANALYSIS SEQUENCE
# ==============================================================================

def _print_flight(name: str, res: SimResult) -> None:
    b = res.burnout
    print(f"\n  {name}")
    print(f"    burnout   : t = {b['t']:6.1f} s   alt = {b['alt'] / 1e3:7.2f} km"
          f"   V = {b['speed']:7.1f} m/s  (M {b['mach']:.2f})")
    print(f"    max-Q     : {res.max_q / 1e3:6.2f} kPa @ t = {res.t_max_q:5.1f} s"
          " (ascent)")
    print(f"    apogee    : {res.apogee_alt / 1e3:7.2f} km @ t = {res.t_apogee:6.1f} s")
    if res.t_impact:
        print(f"    impact    : t = {res.t_impact:6.1f} s   "
              f"downrange = {res.downrange_km:7.1f} km   "
              f"(reentry peak Q {res.reentry_max_q / 1e3:.0f} kPa)")


def main() -> None:
    print("=" * 78)
    print(" 6-DOF ROCKET FLIGHT DYNAMICS SIMULATOR")
    print("=" * 78)

    print("\n[1/5] Model self-verification")
    _self_test()

    veh = Vehicle()
    prm = SimParams()
    twr = veh.thrust_sl / (veh.m_wet * G0)
    print(f"    vehicle: {veh.m_wet / 1e3:.0f} t wet / {veh.m_dry / 1e3:.0f} t dry, "
          f"{veh.thrust_sl / 1e3:.0f} kN, Isp {veh.isp:.0f} s, "
          f"burn {veh.t_burn:.0f} s, lift-off T/W = {twr:.2f}")

    print("\n[2/5] Nominal 6-DOF flight")
    t0 = time.perf_counter()
    nom = simulate(veh, prm, mode="full")
    print(f"    integrated in {time.perf_counter() - t0:.1f} s")
    _print_flight("NOMINAL", nom)

    print("\n[3/5] Pitchover-program optimization "
          f"(max insertion velocity s.t. ascent max-Q <= "
          f"{Q_LIMIT_PA / 1e3:.0f} kPa)")
    opt_info = optimize_pitchover(veh, prm)
    print(f"    SLSQP: {opt_info['message']}  "
          f"({opt_info['nfev']} objective evals, {opt_info['n_sims']} sims, "
          f"{opt_info['elapsed']:.1f} s)")
    print(f"    optimum: t_kick = {opt_info['t_kick']:.2f} s, "
          f"theta_kick = {opt_info['theta_kick_deg']:.2f} deg")
    print(f"    -> V_burnout {opt_info['v_burnout']:.1f} m/s,  "
          f"apogee {opt_info['apogee_km']:.2f} km,  "
          f"max-Q {opt_info['max_q_kpa']:.2f} kPa "
          f"(limit {Q_LIMIT_PA / 1e3:.0f} kPa)")
    prm_opt = replace(prm, t_kick=opt_info["t_kick"],
                      theta_kick=math.radians(opt_info["theta_kick_deg"]))
    opt = simulate(veh, prm_opt, mode="full")
    _print_flight("OPTIMIZED", opt)

    print(f"\n[4/5] Monte Carlo dispersion analysis (N = {N_MONTE_CARLO})")
    mc = run_monte_carlo(veh, prm)
    ok = ~np.isnan(mc["impact_e"])
    print(f"    apogee    : {mc['apogee'].mean() / 1e3:7.2f} km  "
          f"+- {mc['apogee'].std() / 1e3:5.2f} km (1-sigma)")
    print(f"    max-Q     : {mc['max_q'].mean() / 1e3:7.2f} kPa "
          f"+- {mc['max_q'].std() / 1e3:5.2f} kPa")
    print(f"    downrange : {np.nanmean(mc['downrange']):7.1f} km  "
          f"+- {np.nanstd(mc['downrange']):5.1f} km")
    print(f"    impact dispersion: sigma_E = {mc['impact_e'][ok].std():.1f} km, "
          f"sigma_N = {mc['impact_n'][ok].std():.1f} km "
          f"({int(ok.sum())}/{N_MONTE_CARLO} impacts before t_max)")

    print("\n[5/5] Rendering analysis plots")
    _apply_style()
    plot_trajectory_3d(nom, opt, "rocketsim_trajectory3d.png")
    plot_dashboard(nom, "rocketsim_dashboard.png")
    plot_monte_carlo(mc, nom, "rocketsim_montecarlo.png")

    print("\nDone.")
    plt.show()


if __name__ == "__main__":
    main()
