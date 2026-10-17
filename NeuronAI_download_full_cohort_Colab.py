# =============================================================================
# NeuronAI v2.0 - process the COMPLETE Duville et al. datasets in Google Colab
# (the EEG files never pass through your computer).
#
# Mendeley Data protects its pages with a browser check, so the download link
# must be obtained once in your own browser:
#   1. Open https://data.mendeley.com/datasets/7zf99hxxc9/1 (ASD) and click "Download All".
#   2. As soon as the download starts, open the downloads page of your browser
#      (Chrome: chrome://downloads ; Edge: edge://downloads ; Firefox: Ctrl+Shift+Y),
#      right-click the file -> "Copy link address" and paste it below in DIRECT_URLS["ASD"].
#      You can then cancel the download on your computer.
#   3. Repeat with https://data.mendeley.com/datasets/spwnt8t25y/1 (TD) -> DIRECT_URLS["TD"].
#   4. Paste this whole file into ONE Colab cell and press Run. Keep the tab open
#      (about 30-60 min). At the end "NeuronAI_full_cohort_features.zip" (a few MB)
#      is downloaded: that small file is all that is needed for the rest of the analysis.
# The links expire after a while; if the download fails with 403/expired, copy fresh links.
# Alternative: put the two zip files in Google Drive and use MANUAL_ZIPS instead.
# =============================================================================
DIRECT_URLS = {
    "ASD": "",                   # paste the ASD "Download All" link here (between the quotes)
    "TD":  "",                   # paste the TD "Download All" link here
}
MANUAL_ZIPS = {}                 # alternative, e.g. {"ASD": "/content/drive/MyDrive/ASD.zip", "TD": "/content/drive/MyDrive/TD.zip"}
DATASETS = {"ASD": ("7zf99hxxc9", 1), "TD": ("spwnt8t25y", 1)}   # Mendeley identifiers (for the record)
BLOCK = 19                       # recordings per child in sequential numbering (1 rest + 18 task)
SAVE_TO_DRIVE = False            # True: also copy the result zip to Google Drive (MyDrive/NeuronAI)

# ----------------------------------------------------------------------------- source code (plain text)
SOURCE_FILES = {}
SOURCE_FILES['neuronai/__init__.py'] = r'''
'''

SOURCE_FILES['neuronai/model.py'] = r'''"""
neuronai.model
==============

Multi-column extension of the Wendling et al. (2002) neural mass model with

* differentiated glutamatergic transmission (AMPA, NMDA-GluN2A, NMDA-GluN2B),
  including a voltage-dependent Mg2+ block (Jahr & Stevens, 1990);
* NMDA drive onto fast (somatic-targeting, PV-like) interneurons;
* neuromodulatory gain control (dopamine, serotonin, noradrenaline) implemented
  as explicit, bounded transformations of the physiological parameters;
* a network of N cortical columns placed under the EEG electrodes, coupled
  through distance-dependent long-range excitatory connections, and observed
  through a fixed volume-conduction (spatial mixing) operator.

All post-synaptic potential (PSP) kernels are second-order linear filters
h(t) = H k t exp(-k t).  They are integrated with the *exact* discretisation of
the linear part under a zero-order hold on the input (exponential integrator),
which is unconditionally stable for the fast GABA_A kernel (g = 500 s^-1).

The simulator is vectorised over a batch of parameter sets, so thousands of
simulations can be generated efficiently with NumPy only (Colab-compatible).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.signal import decimate

# --------------------------------------------------------------------------
# Montage: the 17 electrodes shared by the two acquisition systems
# --------------------------------------------------------------------------
EEG_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8']

_RAW_POS = {
    'Fp1': (-0.309, 0.951, 0.0), 'Fp2': (0.309, 0.951, 0.0),
    'F7': (-0.809, 0.588, 0.0), 'F3': (-0.50, 0.67, 0.55),
    'Fz': (0.0, 0.719, 0.695), 'F4': (0.50, 0.67, 0.55),
    'F8': (0.809, 0.588, 0.0), 'T7': (-1.0, 0.0, 0.0),
    'C3': (-0.719, 0.0, 0.695), 'Cz': (0.0, 0.0, 1.0),
    'C4': (0.719, 0.0, 0.695), 'T8': (1.0, 0.0, 0.0),
    'P7': (-0.809, -0.588, 0.0), 'P3': (-0.50, -0.67, 0.55),
    'Pz': (0.0, -0.719, 0.695), 'P4': (0.50, -0.67, 0.55),
    'P8': (0.809, -0.588, 0.0)}


def electrode_positions(channels: Sequence[str] = EEG_CHANNELS) -> np.ndarray:
    """Approximate unit-sphere coordinates of 10-20 electrodes."""
    pos = np.array([_RAW_POS[c] for c in channels], dtype=float)
    return pos / np.linalg.norm(pos, axis=1, keepdims=True)


def connectivity_matrices(channels: Sequence[str] = EEG_CHANNELS,
                          coupling_length: float = 0.5,
                          mixing_width: float = 0.30):
    """Return (W, M): row-normalised long-range coupling and volume-conduction mixing."""
    pos = electrode_positions(channels)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    W = np.exp(-d / coupling_length)
    np.fill_diagonal(W, 0.0)
    W /= W.sum(axis=1, keepdims=True)
    M = np.exp(-d ** 2 / (2.0 * mixing_width ** 2))
    M /= M.sum(axis=1, keepdims=True)
    return W, M


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
# Subject-level (fitted) physiological parameters and their prior bounds.
FIT_PARAMS = ['A', 'B', 'G', 'N', 'f2B', 'p', 'K', 'sigma', 'rho']
FIT_BOUNDS = {
    'A': (2.6, 4.6),      # mV, AMPA (fast excitatory) PSP gain
    'B': (10.0, 50.0),    # mV, slow dendritic inhibitory PSP gain
    'G': (5.0, 40.0),     # mV, fast somatic (GABA_A) inhibitory PSP gain
    'N': (0.0, 3.0),      # mV, total NMDA PSP gain on pyramidal cells
    'f2B': (0.0, 1.0),    # fraction of NMDA current carried by GluN2B
    'p': (60.0, 260.0),   # s^-1, mean extrinsic (thalamo-cortical) drive
    'K': (0.0, 1.5),      # long-range glutamatergic coupling strength
    'sigma': (10.0, 60.0),  # s^-1, s.d. of the stochastic drive
    'rho': (0.0, 0.95),   # fraction of drive variance carried by slow (OU, tau = 0.5 s) fluctuations
}
FIT_LABELS = {
    'A': 'AMPA gain $A$', 'B': 'Slow inhibition $B$', 'G': 'Fast GABA$_A$ $G$',
    'N': 'NMDA gain $N$', 'f2B': 'GluN2B fraction $f_{2B}$',
    'p': 'Tonic drive $\\bar p$', 'K': 'Coupling $K$', 'sigma': 'Noise $\\sigma$',
    'rho': 'Slow-input fraction $\\rho$'}


@dataclass
class ModelConstants:
    """Fixed constants (Jansen & Rit, 1995; Wendling et al., 2002)."""
    a: float = 100.0      # s^-1, AMPA rate
    b: float = 50.0       # s^-1, slow inhibitory rate
    g: float = 500.0      # s^-1, fast inhibitory rate
    a2A: float = 1.0 / 0.050   # s^-1, GluN2A (tau ~ 50 ms)
    a2B: float = 1.0 / 0.250   # s^-1, GluN2B (tau ~ 250 ms)
    C: float = 135.0
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    mg: float = 1.0       # mM extracellular Mg2+
    v_rest: float = -65.0  # mV
    NI_ratio: float = 0.5  # NMDA gain on fast interneurons relative to N
    coupling_scale: float = 50.0
    tau_slow: float = 0.5   # s, correlation time of the slow input component
    mixing_width: float = 0.50
    fs_sim: float = 1000.0
    fs_out: float = 250.0


# Pharmacological control knobs (u in [-1, 1]; +1 = maximal agonism/potentiation)
CONTROLS = ['GABA_A', 'GABA_slow', 'AMPA', 'NMDA_2A', 'NMDA_2B',
            'Drive', 'DA', '5HT', 'NE']
CONTROL_LABELS = {
    'GABA_A': 'GABA$_A$ (fast)', 'GABA_slow': 'Slow/dendritic GABA',
    'AMPA': 'AMPA', 'NMDA_2A': 'NMDA-GluN2A', 'NMDA_2B': 'NMDA-GluN2B',
    'Drive': 'Tonic drive', 'DA': 'Dopamine (D1)', '5HT': 'Serotonin (5-HT$_{2A}$)',
    'NE': 'Noradrenaline'}
MAX_REL_CHANGE = 0.30   # +/-30 % maximal change of a receptor gain


def apply_controls(theta: Dict[str, np.ndarray], u: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Map pharmacological controls u (..., 9) onto physiological parameters.

    Receptor-level knobs scale the corresponding gain by (1 + 0.3 u).
    Neuromodulators act through documented gain-control mechanisms:

    * DA (D1-like): potentiates NMDA currents in pyramidal cells (+20 %) and
      increases fast-spiking interneuron excitability (+10 % G)
      (Seamans & Yang, 2004; Gorelova et al., 2002).
    * 5-HT (5-HT2A): depolarises pyramidal neurons, i.e. lowers the firing
      threshold v0 by up to 1 mV, and recruits fast interneurons (+5 % G)
      (Aghajanian & Marek, 1997; Puig & Gulledge, 2011).
    * NE: increases the slope (gain) of the firing-rate function by up to 15 %
      and reduces the stochastic drive by up to 20 % (signal-to-noise
      enhancement; Servan-Schreiber et al., 1990; Aston-Jones & Cohen, 2005).
    """
    out = {k: np.asarray(v, dtype=float).copy() for k, v in theta.items()}
    out.setdefault('dv0', np.zeros_like(out['A']))
    out.setdefault('rscale', np.ones_like(out['A']))
    if u is None:
        return out
    u = np.atleast_2d(np.asarray(u, dtype=float))
    u = np.clip(u, -1.0, 1.0)
    shp = out['A'].shape
    col = lambda i: u[:, i].reshape(shp) if u.shape[0] == np.prod(shp) else u[0, i] * np.ones(shp)
    m = MAX_REL_CHANGE
    out['G'] = out['G'] * (1 + m * col(0)) * (1 + 0.10 * col(6)) * (1 + 0.05 * col(7))
    out['B'] = out['B'] * (1 + m * col(1))
    out['A'] = out['A'] * (1 + m * col(2))
    n2a = out['N'] * (1 - out['f2B']) * (1 + m * col(3))
    n2b = out['N'] * out['f2B'] * (1 + m * col(4))
    n2a = n2a * (1 + 0.20 * col(6))
    n2b = n2b * (1 + 0.20 * col(6))
    out['N'] = n2a + n2b
    out['f2B'] = np.where(out['N'] > 0, n2b / np.maximum(out['N'], 1e-12), out['f2B'])
    out['p'] = out['p'] * (1 + m * col(5))
    out['dv0'] = out['dv0'] - 1.0 * col(7)
    out['rscale'] = out['rscale'] * (1 + 0.15 * col(8))
    out['sigma'] = out['sigma'] * (1 - 0.20 * col(8))
    return out


def _psp_discretisation(k: float, dt: float):
    """Exact ZOH discretisation of y'' = k*x - 2k y' - k^2 y (unit gain)."""
    Am = np.array([[0.0, 1.0], [-k * k, -2.0 * k]])
    Bm = np.array([[0.0], [k]])
    Mx = np.zeros((3, 3))
    Mx[:2, :2] = Am
    Mx[:2, 2:] = Bm
    E = expm(Mx * dt)
    return E[:2, :2], E[:2, 2]


def _psp_discretisation_vec(k: np.ndarray, dt: float):
    """Closed-form exact ZOH discretisation for an array of rate constants k (B,1)."""
    e = np.exp(-k * dt)
    P = ((1 + k * dt) * e, dt * e, -k * k * dt * e, (1 - k * dt) * e)
    Gv = ((1 - e * (1 + k * dt)) / k, k * dt * e)
    return P, Gv


def params_from_unit(z: np.ndarray, names: Sequence[str] = FIT_PARAMS) -> Dict[str, np.ndarray]:
    """Convert unit-cube coordinates (..., P) into a parameter dictionary."""
    z = np.atleast_2d(z)
    return {n: FIT_BOUNDS[n][0] + z[:, i] * (FIT_BOUNDS[n][1] - FIT_BOUNDS[n][0])
            for i, n in enumerate(names)}


def params_to_unit(theta: Dict[str, np.ndarray], names: Sequence[str] = FIT_PARAMS) -> np.ndarray:
    cols = [(np.asarray(theta[n], float) - FIT_BOUNDS[n][0]) /
            (FIT_BOUNDS[n][1] - FIT_BOUNDS[n][0]) for n in names]
    return np.stack(cols, axis=-1)


DEFAULT_THETA = {'A': 3.5, 'B': 30.0, 'G': 20.0, 'N': 1.0, 'f2B': 0.4,
                 'p': 140.0, 'K': 0.3, 'sigma': 30.0, 'rho': 0.5}


@dataclass
class SimulationResult:
    eeg: np.ndarray            # (batch, channels, samples) sensor-level, common-average referenced
    fs: float
    channels: list = field(default_factory=lambda: list(EEG_CHANNELS))


def simulate(theta: Dict[str, np.ndarray], u: Optional[np.ndarray] = None,
             duration: float = 20.0, transient: float = 2.0, seed: int = 0,
             const: ModelConstants = ModelConstants(),
             channels: Sequence[str] = EEG_CHANNELS,
             return_sources: bool = False, crn_index: Optional[np.ndarray] = None) -> SimulationResult:
    """Simulate a batch of networks of extended Wendling columns.

    Parameters
    ----------
    theta : dict of arrays of shape (batch,) (scalars are broadcast).
    u : optional (batch, 9) array of pharmacological controls in [-1, 1].
    duration : seconds of retained signal; transient : discarded warm-up.
    seed : seed of the dedicated NumPy Generator (bit-wise reproducible).
    crn_index : optional integer array (batch,) assigning each row to a noise stream; rows that
        share an index receive identical noise (common random numbers for paired comparisons).
    """
    th = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in theta.items()}
    bsz = max(v.shape[0] for v in th.values())
    th = {k: np.broadcast_to(v, (bsz,)).copy() for k, v in th.items()}
    th = apply_controls(th, u)
    rng = np.random.default_rng(seed)
    c = const
    dt = 1.0 / c.fs_sim
    n_ch = len(channels)
    W, Mmix = connectivity_matrices(channels, mixing_width=c.mixing_width)

    col = lambda x: x[:, None]  # (B,) -> (B,1) broadcast over columns
    A, B, G = col(th['A']), col(th['B']), col(th['G'])
    N2A = col(th['N'] * (1.0 - th['f2B']))
    N2B = col(th['N'] * th['f2B'])
    NI = col(th['N'] * c.NI_ratio * th.get('NI_scale', np.ones(bsz)))   # NMDA on fast interneurons
    pbar, sig, K = col(th['p']), col(th['sigma']), col(th['K'])
    rho = col(th.get('rho', np.full(bsz, DEFAULT_THETA['rho'])))
    sw, ss = sig * np.sqrt(1.0 - rho), sig * np.sqrt(rho)   # white / slow s.d.
    phi = np.exp(-dt / c.tau_slow)
    sq = np.sqrt(1.0 - phi ** 2)
    v0p = c.v0 + col(th['dv0'])
    rr = c.r * col(th['rscale'])
    C1, C2, C3, C4 = c.C, 0.8 * c.C, 0.25 * c.C, 0.25 * c.C
    C5, C6, C7 = 0.3 * c.C, 0.1 * c.C, 0.8 * c.C
    two_e0 = 2.0 * c.e0

    def S(v, v0=c.v0):
        return two_e0 / (1.0 + np.exp(rr * (v0 - v)))

    Pa, Ga_ = _psp_discretisation(c.a, dt)
    Pb, Gb_ = _psp_discretisation(c.b, dt)
    g_tau = col(th['g_tau']) if 'g_tau' in th else np.ones((bsz, 1))
    Pg_v, Gg_v = _psp_discretisation_vec(c.g / g_tau, dt)   # GABA_A decay prolongation (g_tau > 1)
    P2a, G2a_ = _psp_discretisation(c.a2A, dt)
    P2b, G2b_ = _psp_discretisation(c.a2B, dt)

    shape = (bsz, n_ch)
    # state: (value, derivative) for each PSP
    y0 = np.zeros(shape); z0 = np.zeros(shape)   # pyramidal -> interneurons (AMPA)
    y1 = np.zeros(shape); z1 = np.zeros(shape)   # excitatory input to pyramidal (AMPA)
    y2 = np.zeros(shape); z2 = np.zeros(shape)   # slow dendritic inhibition
    y3 = np.zeros(shape); z3 = np.zeros(shape)   # fast somatic inhibition
    y4 = np.zeros(shape); z4 = np.zeros(shape)   # slow -> fast inhibition
    na = np.zeros(shape); za = np.zeros(shape)   # NMDA GluN2A on pyramidal
    nb = np.zeros(shape); zb = np.zeros(shape)   # NMDA GluN2B on pyramidal
    ni = np.zeros(shape); zi = np.zeros(shape)   # NMDA on fast interneurons
    if crn_index is not None:
        crn_index = np.asarray(crn_index, int)
        n_streams = int(crn_index.max()) + 1
        draw = lambda: rng.standard_normal((n_streams, n_ch))[crn_index]
    else:
        draw = lambda: rng.standard_normal(shape)
    eta = draw()                                  # slow OU input (unit variance)

    n_tr = int(round(transient * c.fs_sim))
    n_keep = int(round(duration * c.fs_sim))
    out = np.empty((bsz, n_ch, n_keep), dtype=np.float32)
    mg_k = c.mg / 3.57

    def step(y, z, P, Gv, inp):
        return P[0, 0] * y + P[0, 1] * z + Gv[0] * inp, P[1, 0] * y + P[1, 1] * z + Gv[1] * inp

    for it in range(n_tr + n_keep):
        vp = y1 + na + nb - y2 - y3
        Sp = S(vp, v0p)
        Se = S(C1 * y0)
        Ss = S(C3 * y0)
        Sf = S(C5 * y0 - C6 * y4 + C5 * ni)
        mgb = 1.0 / (1.0 + mg_k * np.exp(-0.062 * (c.v_rest + vp)))
        eta = phi * eta + sq * draw()
        drive = pbar + sw * draw() + ss * eta + K * c.coupling_scale * (Sp @ W.T)
        exc = C2 * Se
        y0, z0 = step(y0, z0, Pa, Ga_, A * Sp)
        y1, z1 = step(y1, z1, Pa, Ga_, A * (drive + exc))
        y2, z2 = step(y2, z2, Pb, Gb_, B * C4 * Ss)
        inp3 = G * C7 * Sf
        y3, z3 = (Pg_v[0] * y3 + Pg_v[1] * z3 + Gg_v[0] * inp3, Pg_v[2] * y3 + Pg_v[3] * z3 + Gg_v[1] * inp3)
        y4, z4 = step(y4, z4, Pb, Gb_, B * Ss)
        na, za = step(na, za, P2a, G2a_, N2A * mgb * exc * 0.1)
        nb, zb = step(nb, zb, P2b, G2b_, N2B * mgb * exc * 0.1)
        ni, zi = step(ni, zi, P2a, G2a_, NI * Sp)
        if it >= n_tr:
            out[:, :, it - n_tr] = vp
    if return_sources:
        src = out.copy()
    # volume conduction + common average reference
    sens = np.einsum('ij,bjt->bit', Mmix, out)
    sens -= sens.mean(axis=1, keepdims=True)
    sens -= sens.mean(axis=2, keepdims=True)
    q = int(round(c.fs_sim / c.fs_out))
    sens = decimate(sens, q, ftype='fir', axis=-1, zero_phase=True).astype(np.float32)
    res = SimulationResult(eeg=sens, fs=c.fs_out, channels=list(channels))
    if return_sources:
        res.sources = src
    return res


def theta_dict(**kw) -> Dict[str, float]:
    d = dict(DEFAULT_THETA)
    d.update(kw)
    return d
'''

SOURCE_FILES['neuronai/features.py'] = r'''"""
neuronai.features
=================

Unit-invariant EEG descriptors shared by simulated and empirical data.

Every feature is invariant to a global rescaling of the signal (microvolts,
volts, or model millivolts give identical values), which removes the
unit-mismatch problem of NeuronAI v1.0.

Features (FEATURE_NAMES):
  rel_delta, rel_theta, rel_alpha, rel_beta              log10 relative power (1-30 Hz)
  iaf                                                    individual alpha frequency (Hz)
  exponent                                               aperiodic 1/f exponent
  plv_alpha, wpli_alpha, kuramoto_alpha                  alpha-band phase synchrony
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert

BANDS = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
         'beta': (13.0, 30.0)}
FMIN, FMAX = 1.0, 30.0   # fitting range; scalp gamma is dominated by EMG/device noise
FEATURE_NAMES = ['rel_delta', 'rel_theta', 'rel_alpha', 'rel_beta',
                 'iaf', 'exponent', 'plv_alpha', 'wpli_alpha', 'kuramoto_alpha']
FEATURE_LABELS = {
    'rel_delta': r'$\delta$ rel. power', 'rel_theta': r'$\theta$ rel. power',
    'rel_alpha': r'$\alpha$ rel. power', 'rel_beta': r'$\beta$ rel. power',
    'iaf': 'IAF (Hz)',
    'exponent': '1/f exponent', 'plv_alpha': r'PLV$_\alpha$',
    'wpli_alpha': r'wPLI$_\alpha$', 'kuramoto_alpha': r'Kuramoto $r_\alpha$'}


def psd(x: np.ndarray, fs: float, win_sec: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD along the last axis (Hann, 2-s windows, 50 % overlap)."""
    nper = int(round(win_sec * fs))
    f, P = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1, detrend='constant')
    return f, P


def normalised_log_psd(f: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Channel-averaged relative PSD (sums to 1 in 1-45 Hz), log10. Shape (..., n_freq)."""
    sel = (f >= FMIN) & (f <= FMAX)
    Pm = P[..., sel].mean(axis=-2) if P.ndim >= 2 else P[sel]
    Pm = Pm / Pm.sum(axis=-1, keepdims=True)
    return f[sel], np.log10(Pm + 1e-20)


def spectral_features(f: np.ndarray, P: np.ndarray) -> Dict[str, np.ndarray]:
    """P: (..., channels, freqs). Returns dict of arrays with shape (...)."""
    sel = (f >= FMIN) & (f <= FMAX)
    Pm = P.mean(axis=-2)  # average over channels
    tot = Pm[..., sel].sum(axis=-1)
    out = {}
    for name, (lo, hi) in BANDS.items():
        idx = (f >= lo) & (f < hi) if name != 'beta' else (f >= lo) & (f <= hi)
        out['rel_' + name] = np.log10(Pm[..., idx].sum(axis=-1) / tot + 1e-20)
    # individual alpha frequency: centre of gravity in 7-14 Hz
    ia = (f >= 7.0) & (f <= 14.0)
    w = Pm[..., ia]
    out['iaf'] = (w * f[ia]).sum(axis=-1) / w.sum(axis=-1)
    # aperiodic exponent: log-log least-squares fit on 2-6 and 15-30 Hz (excludes the alpha peak)
    fa = ((f >= 2.0) & (f <= 6.0)) | ((f >= 15.0) & (f <= 30.0))
    lx = np.log10(f[fa])
    ly = np.log10(Pm[..., fa] + 1e-30)
    lxm = lx - lx.mean()
    slope = ((ly - ly.mean(axis=-1, keepdims=True)) * lxm).sum(axis=-1) / (lxm ** 2).sum()
    out['exponent'] = -slope
    return out


def phase_sync(x: np.ndarray, fs: float, band=(8.0, 13.0), trim_sec: float = 0.5) -> Dict[str, np.ndarray]:
    """Alpha-band PLV, debiased wPLI and Kuramoto order parameter.

    x : (..., channels, samples).  Returns arrays of shape (...).
    PLV and wPLI are averaged over all channel pairs; the Kuramoto order
    parameter r(t) = |mean_k exp(i phi_k(t))| is averaged over time.
    """
    sos = butter(4, band, btype='band', fs=fs, output='sos')
    xf = sosfiltfilt(sos, x, axis=-1)
    trim = int(trim_sec * fs)
    z = hilbert(xf, axis=-1)[..., trim:-trim]
    ph = np.exp(1j * np.angle(z))
    n_ch = x.shape[-2]
    iu = np.triu_indices(n_ch, 1)
    # PLV via phase-normalised cross products
    T = ph.shape[-1]
    cross = np.einsum('...it,...jt->...ij', ph, ph.conj()) / T
    plv = np.abs(cross)[..., iu[0], iu[1]].mean(axis=-1)
    # debiased weighted phase-lag index (Vinck et al., 2011) on analytic cross-spectra
    zi = z[..., :, None, :]
    zj = z[..., None, :, :]
    # memory-light computation pair by pair
    wpli_vals = []
    for i, j in zip(*iu):
        im = np.imag(z[..., i, :] * np.conj(z[..., j, :]))
        s = im.sum(axis=-1)
        sa = np.abs(im).sum(axis=-1)
        s2 = (im ** 2).sum(axis=-1)
        num = s ** 2 - s2
        den = sa ** 2 - s2
        wpli_vals.append(np.where(den > 0, num / np.maximum(den, 1e-30), 0.0))
    del zi, zj
    wpli = np.mean(np.stack(wpli_vals, axis=-1), axis=-1)
    r = np.abs(ph.mean(axis=-2)).mean(axis=-1)
    return {'plv_alpha': plv, 'wpli_alpha': wpli, 'kuramoto_alpha': r}


def extract_features(x: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    """Full feature dictionary for x of shape (..., channels, samples)."""
    f, P = psd(x, fs)
    feats = spectral_features(f, P)
    feats.update(phase_sync(x, fs))
    return feats


def feature_matrix(feats: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.atleast_1d(feats[k]) for k in FEATURE_NAMES], axis=-1)
'''

SOURCE_FILES['neuronai/data.py'] = r'''"""
neuronai.data
=============

Dependency-light loaders and a fully specified preprocessing pipeline for
scalp EEG (EEGLAB .set/.fdt, EDF/BDF, MATLAB .mat, CSV/TXT).  MNE is used when
available (e.g. on Google Colab) but is not required.

Pipeline (identical for every recording, unit-invariant):
 1. channel selection (17 electrodes common to both montages; legacy names
    T3/T4/T5/T6 mapped to T7/T8/P7/P8);
 2. polyphase resampling to 250 Hz;
 3. zero-phase 1-45 Hz band-pass (4th-order Butterworth, second-order sections),
    60 Hz notch (Mexican mains frequency) and automatic notching of narrowband
    instrumental lines (>= 8 dB above the local spectrum, <= 0.5 Hz wide);
 4. bad-channel detection (robust z of log-variance > 4, or flat) and
    distance-weighted spatial interpolation;
 5. common average reference;
 6. 2-s epoching and rejection of epochs whose log-variance or peak-to-peak
    amplitude is a robust outlier (> median + 3.5 MAD);
 7. concatenation of the clean epochs into 20-s analysis windows.
"""
from __future__ import annotations

import os
import re
import struct
from math import gcd
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly, butter, sosfiltfilt, iirnotch, filtfilt

from .model import EEG_CHANNELS, electrode_positions

FS_TARGET = 250.0
MAX_BAD = 4            # maximum number of interpolated channels (of 17)
MAX_MEDCORR = 0.95     # reference-failure threshold
MIN_CLEAN_SEC = 60.0   # minimum clean data retained
ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}


def _norm_name(ch: str) -> str:
    c = re.sub(r'^(EEG[\s_-]*)', '', str(ch).strip(), flags=re.I)
    c = re.sub(r'[-_](REF|LE|A1|A2|M1|M2|AVG|CZ)$', '', c, flags=re.I)
    c = c.strip()
    for std in EEG_CHANNELS + list(ALIASES.keys()):
        if c.lower() == std.lower():
            return ALIASES.get(std, std)
    return c


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------
def read_edf(path: str) -> Tuple[np.ndarray, float, List[str]]:
    """Minimal EDF/EDF+ reader (16-bit). Returns data (ch, samples) in physical units."""
    with open(path, 'rb') as f:
        hdr = f.read(256)
        n_rec = int(hdr[236:244].decode().strip())
        dur = float(hdr[244:252].decode().strip())
        ns = int(hdr[252:256].decode().strip())
        def field(width):
            raw = f.read(width * ns).decode('latin-1')
            return [raw[i * width:(i + 1) * width].strip() for i in range(ns)]
        labels = field(16); field(80); field(8)
        pmin = np.array(field(8), float); pmax = np.array(field(8), float)
        dmin = np.array(field(8), float); dmax = np.array(field(8), float)
        field(80); nsamp = np.array(field(8), int); field(32)
        raw = np.frombuffer(f.read(), dtype='<i2')
    rec_len = nsamp.sum()
    n_rec = min(n_rec if n_rec > 0 else len(raw) // rec_len, len(raw) // rec_len)
    raw = raw[:n_rec * rec_len].reshape(n_rec, rec_len)
    gain = (pmax - pmin) / (dmax - dmin)
    offs = pmin - gain * dmin
    keep = [i for i, l in enumerate(labels) if 'annotation' not in l.lower()]
    fs_all = nsamp / dur
    fs = float(np.median(fs_all[keep]))
    data, names = [], []
    start = np.concatenate([[0], np.cumsum(nsamp)])
    for i in keep:
        if abs(fs_all[i] - fs) > 1e-6:
            continue
        sig = raw[:, start[i]:start[i + 1]].reshape(-1).astype(float)
        data.append(sig * gain[i] + offs[i]); names.append(labels[i])
    return np.array(data), fs, names


_GDF_DTYPES = {1: '<i1', 2: '<u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<i8', 8: '<u8',
               16: '<f4', 17: '<f8'}


def read_gdf(path: str) -> Tuple[np.ndarray, float, List[str]]:
    """Reader for GDF 1.x files (as exported by g.tec g.USBamp / Smarting recorders).

    Returns data (channels, samples) in physical units, sampling rate and labels.
    """
    with open(path, 'rb') as f:
        buf = f.read()
    ver = buf[:8].decode('latin-1')
    if not ver.startswith('GDF 1'):
        raise ValueError(f'Only GDF 1.x is supported by the built-in reader ({ver}); install MNE.')
    hlen = struct.unpack('<q', buf[184:192])[0]
    nrec = struct.unpack('<q', buf[236:244])[0]
    dn, dd = struct.unpack('<II', buf[244:252])
    ns = struct.unpack('<H', buf[252:254])[0]
    o = 256

    def fld(w):
        nonlocal o
        r = [buf[o + i * w:o + (i + 1) * w] for i in range(ns)]
        o += w * ns
        return r
    labels = [x.decode('latin-1').strip('\x00 ') for x in fld(16)]
    fld(80); fld(8)
    pmin = np.array([struct.unpack('<d', x)[0] for x in fld(8)])
    pmax = np.array([struct.unpack('<d', x)[0] for x in fld(8)])
    dmin = np.array([struct.unpack('<q', x)[0] for x in fld(8)], float)
    dmax = np.array([struct.unpack('<q', x)[0] for x in fld(8)], float)
    fld(80)
    spr = np.array([struct.unpack('<i', x)[0] for x in fld(4)])
    typ = np.array([struct.unpack('<I', x)[0] for x in fld(4)])
    if len(set(spr)) != 1 or len(set(typ)) != 1:
        raise ValueError('Mixed sampling rates / data types are not supported by the built-in reader.')
    dt = np.dtype(_GDF_DTYPES[int(typ[0])])
    rec_bytes = int(spr[0]) * ns * dt.itemsize
    avail = (len(buf) - hlen) // rec_bytes
    nrec = avail if nrec < 0 else min(nrec, avail)
    raw = np.frombuffer(buf, dtype=dt, count=nrec * spr[0] * ns, offset=hlen)
    raw = raw.reshape(nrec, ns, spr[0]).transpose(1, 0, 2).reshape(ns, -1).astype(float)
    gain = np.where(dmax > dmin, (pmax - pmin) / np.where(dmax > dmin, dmax - dmin, 1), 1.0)
    data = (raw - dmin[:, None]) * gain[:, None] + pmin[:, None]
    fs = spr[0] / (dn / dd)
    return data, float(fs), labels


def read_set(path: str) -> Tuple[np.ndarray, float, List[str]]:
    """EEGLAB .set reader (data embedded or in a companion .fdt file)."""
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        EEG = m['EEG'] if 'EEG' in m else m
        get = (lambda k: getattr(EEG, k)) if not isinstance(EEG, dict) else (lambda k: EEG[k])
        fs = float(get('srate')); nbchan = int(get('nbchan')); pnts = int(get('pnts'))
        trials = int(get('trials')) if np.size(get('trials')) else 1
        chanlocs = get('chanlocs')
        names = [str(c.labels) for c in np.atleast_1d(chanlocs)]
        dat = get('data')
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py
        with h5py.File(path, 'r') as h:
            g = h['EEG'] if 'EEG' in h else h
            fs = float(np.array(g['srate']).squeeze()); nbchan = int(np.array(g['nbchan']).squeeze())
            pnts = int(np.array(g['pnts']).squeeze()); trials = int(np.array(g['trials']).squeeze())
            refs = g['chanlocs']['labels']
            names = [''.join(chr(c) for c in np.array(h[r[0]]).squeeze()) for r in refs]
            dat = g['data']
            dat = np.array(dat) if not isinstance(dat, (str, bytes)) and dat.dtype != 'uint16' else \
                ''.join(chr(c) for c in np.array(dat).squeeze())
    if isinstance(dat, str):
        fdt = os.path.join(os.path.dirname(path), os.path.basename(dat))
        if not os.path.exists(fdt):
            fdt = os.path.splitext(path)[0] + '.fdt'
        x = np.fromfile(fdt, dtype='<f4').reshape((nbchan, pnts * trials), order='F')
    else:
        x = np.asarray(dat, float).reshape(nbchan, -1, order='F') if np.ndim(dat) == 3 else np.asarray(dat, float)
        if x.shape[0] != nbchan:
            x = x.T
    return x.astype(float), fs, names


def read_any(path: str, fs_hint: Optional[float] = None) -> Tuple[np.ndarray, float, List[str]]:
    ext = os.path.splitext(path)[1].lower()
    try:
        import mne  # noqa: F401  (Colab path)
        mne.set_log_level('ERROR')
        if ext in ('.set', '.edf', '.bdf', '.gdf', '.vhdr', '.fif', '.cnt'):
            raw = mne.io.read_raw(path, preload=True, verbose=False)
            return raw.get_data(), float(raw.info['sfreq']), list(raw.ch_names)
    except Exception:
        pass
    if ext == '.edf':
        return read_edf(path)
    if ext == '.gdf':
        return read_gdf(path)
    if ext == '.set':
        return read_set(path)
    if ext == '.mat':
        m = loadmat(path, squeeze_me=True)
        arrs = {k: v for k, v in m.items() if isinstance(v, np.ndarray) and v.ndim == 2 and min(v.shape) >= 8}
        if not arrs:
            raise ValueError(f'No 2-D EEG array found in {path}')
        k = max(arrs, key=lambda k: arrs[k].size)
        x = arrs[k]
        x = x if x.shape[0] < x.shape[1] else x.T
        fs = fs_hint or float(next((np.squeeze(v) for kk, v in m.items() if kk.lower() in ('fs', 'srate', 'sfreq')), 0) or 0)
        names = list(EEG_CHANNELS[:x.shape[0]])
        return x.astype(float), fs, names
    if ext in ('.csv', '.txt', '.tsv'):
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine='python')
        names = [str(c) for c in df.columns]
        return df.values.T.astype(float), fs_hint, names
    raise ValueError(f'Unsupported format: {path}')


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def select_channels(x, names):
    nm = [_norm_name(n) for n in names]
    idx, missing = [], []
    for ch in EEG_CHANNELS:
        if ch in nm:
            idx.append(nm.index(ch))
        else:
            missing.append(ch)
    if missing:
        raise ValueError(f'Missing channels: {missing}; available: {names}')
    return x[idx]


def _interp_matrix(bad: np.ndarray) -> np.ndarray:
    pos = electrode_positions()
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    Wt = np.where(bad[None, :], 0.0, 1.0 / (d + 1e-6) ** 2)
    np.fill_diagonal(Wt, 0.0)
    Wt /= Wt.sum(1, keepdims=True)
    T = np.eye(len(bad))
    T[bad] = Wt[bad]
    return T


def detect_narrowband(x: np.ndarray, fs: float, thr: float = 0.8, max_width: float = 0.5,
                      fmin: float = 2.0, fmax: float = 45.0) -> List[float]:
    """Centre frequencies of narrowband (non-physiological) spectral lines in the channel-averaged PSD."""
    from scipy.signal import welch, medfilt
    xc = x - x.mean(axis=0, keepdims=True)
    f, P = welch(xc, fs=fs, nperseg=int(8 * fs), axis=-1)
    L = np.log10(P.mean(0) + 1e-30)
    df_ = f[1] - f[0]
    base = medfilt(L, int(4.0 / df_) | 1)
    ex = (L - base) > thr
    ex &= (f >= fmin) & (f <= fmax)
    peaks, i = [], 0
    while i < len(ex):
        if ex[i]:
            j = i
            while j + 1 < len(ex) and ex[j + 1]:
                j += 1
            if (j - i + 1) * df_ <= max_width + 1e-9:
                seg = slice(i, j + 1)
                peaks.append(float(f[seg][np.argmax(L[seg])]))
            i = j + 1
        else:
            i += 1
    return peaks


def preprocess(x: np.ndarray, fs: float, names: List[str], line_freq: float = 60.0,
               epoch_sec: float = 2.0, window_sec: float = 20.0, max_sec: Optional[float] = None):
    """Return dict with clean continuous data (17 x T), windows list, and QC info."""
    x = select_channels(np.asarray(x, float), names)
    x = x - np.median(x, axis=1, keepdims=True)
    # resample
    fs_i = int(round(fs))
    g = gcd(int(FS_TARGET), fs_i)
    x = resample_poly(x, int(FS_TARGET) // g, fs_i // g, axis=1)
    fs = FS_TARGET
    if max_sec:
        x = x[:, :int(max_sec * fs)]
    sos = butter(4, [1.0, 45.0], btype='band', fs=fs, output='sos')
    x = sosfiltfilt(sos, x, axis=1)
    if line_freq < fs / 2:
        bn, an = iirnotch(line_freq, 30.0, fs)
        x = filtfilt(bn, an, x, axis=1)
    # narrowband instrumental artefacts (e.g. a ~16 Hz line present in every recording of one system):
    # peaks >= 0.8 log10 units (8 dB) above the 4-Hz running median and <= 0.5 Hz wide, 2-45 Hz
    line_peaks = detect_narrowband(x, fs)
    for f0 in line_peaks:
        bn, an = iirnotch(f0, max(f0 / 0.5, 5.0), fs)
        x = filtfilt(bn, an, x, axis=1)
    # recording-level QC 1: reference failure (common-mode contamination) -> near-unit
    # median inter-channel correlation of the reference-montage signals
    Cm = np.corrcoef(x)
    medcorr = float(np.median(Cm[np.triu_indices(x.shape[0], 1)]))
    # bad channels, detected after a provisional common average reference
    xc = x - x.mean(axis=0, keepdims=True)
    lv = np.log(xc.var(axis=1) + 1e-30)
    med = np.median(lv); mad = 1.4826 * np.median(np.abs(lv - med)) + 1e-12
    bad = (np.abs(lv - med) / mad > 4.0) | (x.std(axis=1) < 1e-12)
    if 0 < bad.sum() <= MAX_BAD:
        x = _interp_matrix(bad) @ (x - x[~bad].mean(axis=0, keepdims=True))
    x = x - x.mean(axis=0, keepdims=True)
    # epochs
    L = int(epoch_sec * fs)
    n_ep = x.shape[1] // L
    ep = x[:, :n_ep * L].reshape(x.shape[0], n_ep, L).transpose(1, 0, 2)
    lv_e = np.log(ep.var(axis=2).mean(1))
    p2p = np.log(np.ptp(ep, axis=2).max(1))
    def rob(v):
        m = np.median(v); s = 1.4826 * np.median(np.abs(v - m)) + 1e-12
        return (v - m) / s
    good = (rob(lv_e) < 3.5) & (rob(p2p) < 3.5)
    clean = np.concatenate(list(ep[good]), axis=1) if good.any() else x
    W = int(window_sec * fs)
    n_w = clean.shape[1] // W
    windows = [clean[:, i * W:(i + 1) * W] for i in range(n_w)]
    reasons = []
    if medcorr > MAX_MEDCORR:
        reasons.append(f'reference failure (median inter-channel r = {medcorr:.3f})')
    if bad.sum() > MAX_BAD:
        reasons.append(f'{int(bad.sum())} bad channels')
    if clean.shape[1] / fs < MIN_CLEAN_SEC:
        reasons.append(f'only {clean.shape[1] / fs:.0f} s of clean data')
    return {'clean': clean, 'windows': windows, 'fs': fs, 'line_peaks': line_peaks, 'bad_channels': [EEG_CHANNELS[i] for i in np.where(bad)[0]],
            'n_epochs': int(n_ep), 'n_good_epochs': int(good.sum()), 'medcorr': medcorr,
            'clean_sec': clean.shape[1] / fs, 'excluded': bool(reasons), 'reason': '; '.join(reasons)}
'''

SOURCE_FILES['neuronai/bank.py'] = r'''"""
neuronai.bank
=============

Generation of simulation banks used to train the amortised inverse model
(EEG features -> physiological parameters) and the forward emulator
(physiological parameters -> EEG features).

Parameters are drawn with a scrambled Sobol sequence (seeded) over the prior
box FIT_BOUNDS. Bank "I" (inverse) keeps neuromodulatory tone at baseline;
bank "E" (emulator) additionally varies the two neuromodulatory nuisance
parameters (dv0: firing-threshold shift; rscale: firing-rate gain) over the
range reachable by the pharmacological controls.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.stats import qmc

from .model import FIT_PARAMS, FIT_BOUNDS, simulate, params_from_unit
from .features import extract_features, feature_matrix, psd, normalised_log_psd, FEATURE_NAMES

EMU_EXTRA = {'dv0': (-1.0, 1.0), 'rscale': (0.85, 1.15)}


def sample_unit(n: int, d: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n, 2))))
    return sampler.random_base2(m)[:n]


def generate_bank(n: int, kind: str = 'I', seed: int = 2026, batch: int = 256,
                  duration: float = 20.0, transient: float = 2.0,
                  out_path: Optional[str] = None, verbose: bool = True) -> dict:
    d = len(FIT_PARAMS) + (2 if kind == 'E' else 0)
    U = sample_unit(n, d, seed)
    theta = params_from_unit(U[:, :len(FIT_PARAMS)])
    if kind == 'E':
        for j, (k, (lo, hi)) in enumerate(EMU_EXTRA.items()):
            theta[k] = lo + U[:, len(FIT_PARAMS) + j] * (hi - lo)
    X, LP, ok = [], [], []
    t0 = time.time()
    for s in range(0, n, batch):
        th_b = {k: v[s:s + batch] for k, v in theta.items()}
        res = simulate(th_b, duration=duration, transient=transient, seed=seed * 1000 + s // batch)
        x = res.eeg
        good = np.isfinite(x).all(axis=(1, 2)) & (x.std(axis=(1, 2)) > 1e-6)
        x = np.where(good[:, None, None], x, 0.0) + (~good)[:, None, None] * 1e-3 * \
            np.random.default_rng(0).standard_normal(x.shape)
        F = feature_matrix(extract_features(x, res.fs))
        f, P = psd(x, res.fs)
        fsel, lp = normalised_log_psd(f, P)
        good &= np.isfinite(F).all(axis=1)
        X.append(F); LP.append(lp); ok.append(good)
        if verbose:
            el = time.time() - t0
            done = min(s + batch, n)
            print(f'  bank {kind}: {done}/{n} simulations  ({el/60:.1f} min elapsed)', flush=True)
    bank = {'theta_names': np.array(list(theta.keys())),
            'theta': np.stack([theta[k] for k in theta], axis=1),
            'X': np.concatenate(X), 'logpsd': np.concatenate(LP).astype(np.float32),
            'freqs': fsel, 'ok': np.concatenate(ok), 'feature_names': np.array(FEATURE_NAMES),
            'seed': seed, 'duration': duration}
    if out_path:
        np.savez_compressed(out_path, **bank)
    return bank


def load_bank(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
'''

SOURCE_FILES['neuronai/inference.py'] = r'''"""
neuronai.inference
==================

NeuronAI learning components (scikit-learn only, CPU, bit-wise reproducible):

* ``InverseEnsemble`` - amortised simulation-based inference. A deep ensemble of
  multilayer perceptrons maps the 10 EEG descriptors to the 8 subject-level
  physiological parameters (unit-cube scale). Split-conformal calibration on a
  held-out set of simulations yields distribution-free prediction intervals
  with guaranteed marginal coverage (Vovk et al., 2005; Angelopoulos & Bates, 2023).

* ``ForwardEmulator`` - a deep-ensemble surrogate of the simulator that maps the
  physiological parameters (+ neuromodulatory nuisance parameters) to the
  expected EEG descriptors; used for fast intervention search, after which
  every candidate is re-verified with the full stochastic simulator.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from .model import FIT_PARAMS, FIT_BOUNDS
from .bank import EMU_EXTRA


class InverseEnsemble:
    def __init__(self, n_members: int = 5, hidden=(256, 128, 64), alpha: float = 1e-4,
                 seed: int = 2026, max_iter: int = 600):
        self.n_members, self.hidden, self.alpha = n_members, hidden, alpha
        self.seed, self.max_iter = seed, max_iter
        self.members = []
        self.scaler = StandardScaler()
        self.q = None

    def fit(self, X: np.ndarray, Z: np.ndarray, X_cal: Optional[np.ndarray] = None,
            Z_cal: Optional[np.ndarray] = None, coverage: float = 0.90):
        Xs = self.scaler.fit_transform(X)
        self.members = []
        for m in range(self.n_members):
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation='relu', alpha=self.alpha,
                               learning_rate_init=1e-3, batch_size=128, max_iter=self.max_iter,
                               early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                               random_state=self.seed + m)
            net.fit(Xs, Z)
            self.members.append(net)
        if X_cal is not None:
            self.calibrate(X_cal, Z_cal, coverage)
        return self

    def predict(self, X: np.ndarray, return_members: bool = False):
        Xs = self.scaler.transform(np.atleast_2d(X))
        P = np.stack([np.clip(m.predict(Xs), 0, 1) for m in self.members])
        return (P.mean(0), P) if return_members else P.mean(0)

    def calibrate(self, X_cal, Z_cal, coverage=0.90):
        """Split-conformal absolute-residual quantiles per parameter."""
        pred = self.predict(X_cal)
        res = np.abs(pred - Z_cal)
        n = len(res)
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        self.q = np.quantile(res, level, axis=0, method='higher')
        self.coverage = coverage
        return self.q

    def interval(self, X):
        mu = self.predict(X)
        return np.clip(mu - self.q, 0, 1), np.clip(mu + self.q, 0, 1)

    def score(self, X, Z) -> Dict[str, float]:
        pred = self.predict(X)
        return {n: r2_score(Z[:, i], pred[:, i]) for i, n in enumerate(FIT_PARAMS)}


EMU_INPUTS = list(FIT_PARAMS) + list(EMU_EXTRA.keys())


def emu_matrix(theta: Dict[str, np.ndarray]) -> np.ndarray:
    th = {k: np.atleast_1d(np.asarray(v, float)) for k, v in theta.items()}
    n = max(len(v) for v in th.values())
    th.setdefault('dv0', np.zeros(n))
    th.setdefault('rscale', np.ones(n))
    cols = []
    for k in EMU_INPUTS:
        lo, hi = FIT_BOUNDS.get(k, EMU_EXTRA.get(k))
        # allow controlled extrapolation beyond the prior box (e.g. +30 % gains)
        cols.append((np.broadcast_to(th[k], (n,)) - lo) / (hi - lo))
    return np.stack(cols, axis=1)


class ForwardEmulator:
    def __init__(self, n_members: int = 5, hidden=(256, 256, 128), alpha: float = 1e-4,
                 seed: int = 7, max_iter: int = 800):
        self.n_members, self.hidden, self.alpha = n_members, hidden, alpha
        self.seed, self.max_iter = seed, max_iter
        self.members, self.xs, self.ys = [], StandardScaler(), StandardScaler()

    def fit(self, Theta_unit: np.ndarray, Y: np.ndarray):
        Xs = self.xs.fit_transform(Theta_unit)
        Ys = self.ys.fit_transform(Y)
        self.members = []
        for m in range(self.n_members):
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation='relu', alpha=self.alpha,
                               learning_rate_init=1e-3, batch_size=128, max_iter=self.max_iter,
                               early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                               random_state=self.seed + m)
            net.fit(Xs, Ys)
            self.members.append(net)
        return self

    def predict(self, Theta_unit: np.ndarray, return_std: bool = False):
        Xs = self.xs.transform(np.atleast_2d(Theta_unit))
        P = np.stack([self.ys.inverse_transform(m.predict(Xs)) for m in self.members])
        return (P.mean(0), P.std(0)) if return_std else P.mean(0)

    def score(self, Theta_unit, Y, names: Sequence[str]) -> Dict[str, float]:
        P = self.predict(Theta_unit)
        return {n: r2_score(Y[:, i], P[:, i]) for i, n in enumerate(names)}


class SpectralEmulator(ForwardEmulator):
    """Emulator of the full 1-30 Hz relative log-spectrum (59 bins) through a PCA bottleneck."""

    def __init__(self, n_components: int = 12, **kw):
        super().__init__(**kw)
        self.n_components = n_components

    def fit(self, Theta_unit, LP):
        from sklearn.decomposition import PCA
        self.pca = PCA(self.n_components, random_state=0).fit(LP)
        super().fit(Theta_unit, self.pca.transform(LP))
        return self

    def predict(self, Theta_unit, return_std=False):
        P = super().predict(Theta_unit)
        return self.pca.inverse_transform(P)


def _mlp_forward_grad(net, x):
    """Forward pass of a fitted scikit-learn ReLU MLPRegressor with the Jacobian w.r.t. its input."""
    a = x
    J = np.eye(len(x))
    for li, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        h = a @ W + b
        J = J @ W
        if li < len(net.coefs_) - 1:
            mask = (h > 0).astype(float)
            a = h * mask
            J = J * mask[None, :]
        else:
            a = h
    return a, J


def emulator_value_grad(emulator, u_in):
    """Ensemble-mean emulator prediction and Jacobian (n_features x n_inputs) in original units."""
    xs, ys = emulator.xs, emulator.ys
    x = (u_in - xs.mean_) / xs.scale_
    preds, jacs = [], []
    for m in emulator.members:
        a, J = _mlp_forward_grad(m, x)
        preds.append(a); jacs.append(J)
    P = np.mean(preds, 0) * ys.scale_ + ys.mean_
    Jm = (np.mean(jacs, 0) / xs.scale_[:, None]) * ys.scale_[None, :]   # d pred / d u_in  (inputs x outputs)
    if hasattr(emulator, 'pca'):
        P = P @ emulator.pca.components_ + emulator.pca.mean_
        Jm = Jm @ emulator.pca.components_
    return P, Jm.T


def refine_twin(x: np.ndarray, emulator: 'ForwardEmulator', z_starts: np.ndarray, scale: np.ndarray,
                lo: Optional[np.ndarray] = None, hi: Optional[np.ndarray] = None,
                spectrum: Optional[np.ndarray] = None, spec_emulator: Optional['SpectralEmulator'] = None,
                spec_scale: Optional[np.ndarray] = None):
    """Posterior-mode refinement of a digital twin.

    Minimises the standardised discrepancy between emulated and observed data,
        L(z) = sum_j ((F_j(z) - x_j) / s_j)^2  +  (9 / n_f) * sum_f ((S_f(z) - y_f) / t_f)^2,
    i.e. the nine descriptors plus (optionally) the full relative log-spectrum y, the spectral block
    being weighted as much as the descriptor block. z is restricted to [lo, hi] (the split-conformal
    interval of the amortised estimate); L-BFGS-B with exact emulator gradients and several starts.
    Neuromodulatory nuisance inputs are held at baseline (unit-scale value 0.5)."""
    from scipy.optimize import minimize
    P = len(FIT_PARAMS)
    lo = np.zeros(P) if lo is None else np.asarray(lo)
    hi = np.ones(P) if hi is None else np.asarray(hi)
    extra = np.array([0.5, 0.5])
    use_spec = spectrum is not None and spec_emulator is not None
    w_spec = 9.0 / len(spectrum) if use_spec else 0.0

    def f(z):
        u = np.concatenate([z, extra])
        pred, J = emulator_value_grad(emulator, u)
        r = (pred - x) / scale
        val = float((r ** 2).sum())
        g = (2 * (r / scale) @ J)[:P]
        if use_spec:
            sp, Js = emulator_value_grad(spec_emulator, u)
            rs = (sp - spectrum) / spec_scale
            val += w_spec * float((rs ** 2).sum())
            g = g + w_spec * (2 * (rs / spec_scale) @ Js)[:P]
        return val, g
    best = None
    for z0 in np.atleast_2d(z_starts):
        z0 = np.clip(z0, lo, hi)
        res = minimize(f, z0, jac=True, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'maxiter': 300})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)
'''

SOURCE_FILES['neuronai/intervention.py'] = r'''"""
neuronai.intervention
=====================

In silico pharmacological search.

For a subject with observed descriptors x_s and fitted digital twin theta_s,
the post-intervention descriptors under control vector u are predicted with a
residual-preserving (delta) estimator:

    x_hat(u) = x_s + [F(theta_s (+) u) - F(theta_s)]

where F is the forward emulator (search stage) or the full stochastic simulator
averaged over seeds (verification stage). Distance to the normative reference
is the Mahalanobis distance to the typically-developing (TD) feature
distribution (Ledoit-Wolf shrinkage covariance). The search evaluates all
sparse supports of size 1-3 (9 + 36 + 84 = 129 combinations), first on a dense
grid and then by bounded quasi-Newton refinement of the best supports.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .model import CONTROLS, apply_controls, FIT_PARAMS
from .inference import emu_matrix

LAMBDA_L2 = 0.05


class NormativeReference:
    def __init__(self, X_ref: np.ndarray, feature_idx: Optional[List[int]] = None):
        self.idx = feature_idx if feature_idx is not None else list(range(X_ref.shape[1]))
        Xr = X_ref[:, self.idx]
        self.mu = Xr.mean(0)
        self.sd = Xr.std(0, ddof=1)
        lw = LedoitWolf().fit((Xr - self.mu) / self.sd)
        self.P = np.linalg.pinv(lw.covariance_)        # pseudo-inverse: equals the inverse when non-singular

    def distance(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X)[:, self.idx] - self.mu) / self.sd
        return np.sqrt(np.einsum('ni,ij,nj->n', Z, self.P, Z))


def all_supports(max_k: int = 3):
    sup = []
    for k in range(1, max_k + 1):
        sup.extend(combinations(range(len(CONTROLS)), k))
    return sup


def _theta_rows(theta_s: Dict[str, float], U: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    th = {k: np.full(n, float(theta_s[k])) for k in FIT_PARAMS}
    return emu_matrix(apply_controls(th, U))


def emulated_post(x_s, theta_s, U, emulator, base_pred=None):
    U = np.atleast_2d(U)
    if base_pred is None:
        base_pred = emulator.predict(_theta_rows(theta_s, np.zeros((1, len(CONTROLS)))))[0]
    return x_s[None, :] + emulator.predict(_theta_rows(theta_s, U)) - base_pred[None, :]


def search(x_s: np.ndarray, theta_s: Dict[str, float], emulator, ref: NormativeReference,
           lam: float = LAMBDA_L2, max_k: int = 3, n_refine: int = 12,
           grid_pts=(41, 21, 9), precomputed=None) -> Dict:
    """Exhaustive sparse search; returns ranked results for all 129 supports.

    precomputed: optional (supports, U_all, owners, delta_grid) to reuse emulator predictions."""
    base_pred = emulator.predict(_theta_rows(theta_s, np.zeros((1, len(CONTROLS)))))[0]
    d0 = float(ref.distance(x_s[None, :])[0])
    if precomputed is None:
        supports, U_all, own = grid_controls(max_k, grid_pts)
        D = ref.distance(emulated_post(x_s, theta_s, U_all, emulator, base_pred))
    else:
        supports, U_all, own, delta_grid = precomputed
        D = ref.distance(x_s[None, :] + delta_grid)
    J = D + lam * (U_all ** 2).sum(1)
    results = []
    for si, sup in enumerate(supports):
        m = own == si
        j = np.argmin(J[m])
        results.append({'support': sup, 'u': U_all[m][j].copy(), 'J': float(J[m][j]), 'D': float(D[m][j])})
    order = np.argsort([r['J'] for r in results])
    for oi in order[:n_refine]:
        r = results[oi]
        sup = list(r['support'])

        def obj(v):
            U = np.zeros((1, len(CONTROLS))); U[0, sup] = v
            d = ref.distance(emulated_post(x_s, theta_s, U, emulator, base_pred))[0]
            return d + lam * float((v ** 2).sum())
        opt = minimize(obj, r['u'][sup], method='L-BFGS-B', bounds=[(-1, 1)] * len(sup),
                       options={'maxiter': 60})
        if opt.fun < r['J']:
            U = np.zeros(len(CONTROLS)); U[sup] = opt.x
            r['u'], r['J'] = U, float(opt.fun)
            r['D'] = float(ref.distance(emulated_post(x_s, theta_s, U[None], emulator, base_pred))[0])
    for r in results:
        r['improvement'] = 100.0 * (d0 - r['D']) / d0
        r['name'] = '+'.join(CONTROLS[i] for i in r['support'])
    results.sort(key=lambda r: r['J'])
    return {'d0': d0, 'results': results, 'base_pred': base_pred}


def grid_controls(max_k: int = 3, grid_pts=(41, 21, 9)):
    supports = all_supports(max_k)
    rows, owners = [], []
    for si, sup in enumerate(supports):
        g = np.linspace(-1, 1, grid_pts[len(sup) - 1])
        mesh = np.stack(np.meshgrid(*([g] * len(sup)), indexing='ij'), -1).reshape(-1, len(sup))
        U = np.zeros((len(mesh), len(CONTROLS)))
        U[:, list(sup)] = mesh
        rows.append(U); owners.append(np.full(len(mesh), si))
    return supports, np.concatenate(rows), np.concatenate(owners)


def grid_rank(x_s, delta_grid, U_all, own, supports, ref, lam=LAMBDA_L2):
    """Rank supports from precomputed emulator changes on the grid (no refinement)."""
    d0 = float(ref.distance(x_s[None])[0])
    D = ref.distance(x_s[None] + delta_grid)
    J = D + lam * (U_all ** 2).sum(1)
    best = []
    for si, sup in enumerate(supports):
        m = own == si
        j = np.argmin(J[m])
        best.append((float(J[m][j]), si, float(D[m][j])))
    best.sort()
    J0, si, D0 = best[0]
    return {'name': '+'.join(CONTROLS[i] for i in supports[si]), 'support': supports[si],
            'improvement': 100 * (d0 - D0) / d0}


def member_deltas(theta_s, emulator, U_all):
    """Per-member emulator changes F_m(theta (+) u) - F_m(theta) on the control grid."""
    X = emulator.xs.transform(_theta_rows(theta_s, U_all))
    X0 = emulator.xs.transform(_theta_rows(theta_s, np.zeros((1, len(CONTROLS)))))
    out = []
    for m in emulator.members:
        out.append(emulator.ys.inverse_transform(m.predict(X)) - emulator.ys.inverse_transform(m.predict(X0)))
    return np.stack(out)   # (members, grid, features)


def random_null(x_s, theta_s, u_star, emulator, ref, n: int = 500, seed: int = 0):
    """Improvements of random control vectors with the same support size and L2 norm as u_star."""
    rng = np.random.default_rng(seed)
    k = int((np.abs(u_star) > 1e-9).sum())
    norm = np.linalg.norm(u_star)
    U = np.zeros((n, len(CONTROLS)))
    for i in range(n):
        sup = rng.choice(len(CONTROLS), size=max(k, 1), replace=False)
        v = rng.standard_normal(len(sup))
        v = v / np.linalg.norm(v) * norm
        U[i, sup] = np.clip(v, -1, 1)
    d0 = ref.distance(x_s[None])[0]
    D = ref.distance(emulated_post(x_s, theta_s, U, emulator))
    return 100.0 * (d0 - D) / d0


# ---------------------------------------------------------------------------
# Simulator-in-the-loop verification (guards against surrogate exploitation)
# ---------------------------------------------------------------------------
def random_like(U_ref: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random control vectors matched one-to-one to U_ref in support size and Euclidean norm."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(U_ref)
    for i, u in enumerate(U_ref):
        k = max(int((np.abs(u) > 1e-9).sum()), 1)
        norm = np.linalg.norm(u)
        for _ in range(100):
            sup = rng.choice(len(CONTROLS), size=k, replace=False)
            v = rng.standard_normal(k)
            v = v / np.linalg.norm(v) * norm
            if np.all(np.abs(v) <= 1):
                break
        out[i, sup] = np.clip(v, -1, 1)
    return out


def simulate_candidates(theta_s: Dict[str, float], U: np.ndarray, n_streams: int, seed: int,
                        max_rows: int = 200):
    """Simulate the twin under each control vector (rows of U) with n_streams paired noise streams.

    Returns descriptors of shape (n_candidates, n_streams, n_features); stream s receives identical
    noise for every candidate (common random numbers), so differences isolate the intervention."""
    from .model import simulate
    from .features import extract_features, feature_matrix
    n_c = len(U)
    cand = np.repeat(np.arange(n_c), n_streams)
    stream = np.tile(np.arange(n_streams), n_c)
    F = np.empty((len(cand), 9))
    per = max(n_streams, (max_rows // n_streams) * n_streams)
    for s in range(0, len(cand), per):
        sl = slice(s, s + per)
        th = {k: np.full(len(cand[sl]), float(theta_s[k])) for k in FIT_PARAMS}
        r = simulate(th, u=U[cand[sl]], duration=20.0, seed=seed, crn_index=stream[sl])
        F[sl] = feature_matrix(extract_features(r.eeg, r.fs))
    return F.reshape(n_c, n_streams, -1)


def verified_improvement(x_s, F_cand, F_base, ref: NormativeReference):
    """Per-stream and mean improvement (%) of each candidate using residual-preserving deltas."""
    d0 = float(ref.distance(x_s[None])[0])
    delta = F_cand - F_base[None]                       # (cand, streams, feat)
    per = 100 * (d0 - ref.distance((x_s[None, None] + delta).reshape(-1, len(x_s))).reshape(delta.shape[:2])) / d0
    mean = 100 * (d0 - ref.distance(x_s[None] + delta.mean(1))) / d0
    return per, mean, delta
'''

SOURCE_FILES['neuronai/stats.py'] = r'''"""neuronai.stats - statistics used throughout the analyses."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, func=np.mean, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    bs = np.array([func(x[i]) for i in idx])
    lo, hi = np.percentile(bs, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(func(x)), float(lo), float(hi)


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    return d * (1 - 3 / (4 * (na + nb) - 9))


def hedges_g_ci(a, b, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [hedges_g(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    return hedges_g(a, b), *np.percentile(bs, [2.5, 97.5])


def fdr_bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def icc_2_1(x1, x2):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    Y = np.column_stack([x1, x2]).astype(float)
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def icc_ci(x1, x2, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x1)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(icc_2_1(np.asarray(x1)[i], np.asarray(x2)[i]))
    return icc_2_1(x1, x2), *np.nanpercentile(vals, [2.5, 97.5])


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = stats.spearmanr(np.asarray(x)[m], np.asarray(y)[m])
    return float(r), float(p), int(m.sum())


def mannwhitney(a, b):
    return stats.mannwhitneyu(a, b, alternative='two-sided')


def cluster_bootstrap_ci(x, clusters, func=np.mean, n_boot: int = 5000, ci: float = 0.95, seed: int = 0):
    """Two-stage (cluster, then within-cluster) bootstrap for nested data
    (recordings nested within children)."""
    x = np.asarray(x, float); clusters = np.asarray(clusters)
    labs = np.unique(clusters)
    groups = [x[clusters == c] for c in labs]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = np.concatenate([rng.choice(groups[k], len(groups[k])) for k in pick])
        bs.append(func(sample))
    lo, hi = np.percentile(bs, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(func(x)), float(lo), float(hi)


def child_permutation_test(values, groups, clusters, n_perm: int = 20000, seed: int = 0, max_exact: int = 20000):
    """Permutation test of a group difference at the child (cluster) level.
    Statistic: difference of child-mean averages. All relabellings of children are enumerated when there
    are at most `max_exact` of them (exact test); otherwise `n_perm` random relabellings are used
    (Monte Carlo, p = (1 + #extreme) / (1 + n_perm)). Returns (diff, p_two_sided, n_permutations)."""
    from itertools import combinations
    from math import comb
    v = np.asarray(values, float); g = np.asarray(groups); c = np.asarray(clusters)
    labs = np.unique(c)
    means = np.array([v[c == l].mean() for l in labs])
    lab_group = np.array([g[c == l][0] for l in labs])
    G = np.unique(lab_group)
    in_a = lab_group == G[0]
    obs = means[in_a].mean() - means[~in_a].mean()
    k = int(in_a.sum())
    if comb(len(labs), k) <= max_exact:
        stats_ = []
        for cmb in combinations(range(len(labs)), k):
            m = np.zeros(len(labs), bool); m[list(cmb)] = True
            stats_.append(means[m].mean() - means[~m].mean())
        stats_ = np.array(stats_)
        return float(obs), float((np.abs(stats_) >= abs(obs) - 1e-12).mean()), len(stats_)
    rng = np.random.default_rng(seed)
    ext = 0
    for _ in range(n_perm):
        m = np.zeros(len(labs), bool); m[rng.choice(len(labs), k, replace=False)] = True
        ext += abs(means[m].mean() - means[~m].mean()) >= abs(obs) - 1e-12
    return float(obs), float((1 + ext) / (1 + n_perm)), int(n_perm)


def sign_flip_test(x, n_perm: int = 20000, seed: int = 0, stat=np.mean):
    """One-sample (paired) sign-flip permutation test of H0: distribution symmetric about 0.
    `x` holds one value per independent unit (e.g. per child). Returns (statistic, p_two_sided)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    obs = stat(x)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(x)))
    null = np.array([stat(f * x) for f in flips])
    return float(obs), float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (1 + n_perm))


def icc_1(values, clusters):
    """ICC(1) from one-way random-effects ANOVA (unbalanced): share of variance between clusters."""
    v = np.asarray(values, float); c = np.asarray(clusters)
    labs = np.unique(c)
    k_i = np.array([(c == l).sum() for l in labs]); N = len(v); a = len(labs)
    gm = v.mean()
    msb = sum(k * (v[c == l].mean() - gm) ** 2 for k, l in zip(k_i, labs)) / (a - 1)
    msw = sum(((v[c == l] - v[c == l].mean()) ** 2).sum() for l in labs) / (N - a)
    k0 = (N - (k_i ** 2).sum() / N) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def _ledoit_wolf_cov(X):
    """Ledoit-Wolf shrunk covariance computed on standardised data and rescaled
    (the procedure of scikit-learn's discriminant_analysis._cov(X, shrinkage='auto'))."""
    sc = X.std(0)
    sc[sc == 0] = 1.0
    Z = (X - X.mean(0)) / sc
    n, p = Z.shape
    S = Z.T @ Z / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum()
    Z2 = Z ** 2
    b2 = ((Z2.T @ Z2) / n - S ** 2).sum() / n
    shrink = 0.0 if d2 == 0 else min(b2, d2) / d2
    C = (1 - shrink) * S + shrink * mu * np.eye(p)
    return sc[:, None] * C * sc[None, :]


def _lda_fit_predict(X, y, x_new):
    """Shrinkage linear discriminant, equivalent to scikit-learn's
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') with empirical priors, implemented
    directly in NumPy (about 50 times faster, which makes leave-one-out permutation tests feasible).
    Returns the predicted label of every row of x_new."""
    labs, inv = np.unique(y, return_inverse=True)
    pri = np.bincount(inv) / len(y)
    means = np.stack([X[inv == k].mean(0) for k in range(len(labs))])
    cov = sum(pri[k] * _ledoit_wolf_cov(X[inv == k]) for k in range(len(labs)))
    coef = np.linalg.lstsq(cov, means.T, rcond=None)[0].T
    icpt = -0.5 * np.einsum('ij,ij->i', means, coef) + np.log(pri)
    return labs[np.argmax(np.atleast_2d(x_new) @ coef.T + icpt, axis=1)]


def identify_discriminant(M, labels, halves=()):
    """Leave-one-recording-out identification of the person behind each recording with a
    shrinkage linear discriminant (Ledoit-Wolf), fitted only on the training recordings of each fold.

    M       (n, p) features of the full recordings (e.g. twin parameters)
    labels  (n,) person labels
    halves  optional sequence of (n, p) arrays with the same features estimated on independent
            halves of each recording; they are added to the training set of each fold only, so
            that the within-person covariance also reflects estimation noise.
    Returns the proportion of correctly identified recordings."""
    import warnings
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    M = np.asarray(M, float); labels = np.asarray(labels)
    n, hits = len(M), 0
    for k in range(n):
        tr = np.arange(n) != k
        X = np.vstack([M[tr]] + [np.asarray(h, float)[tr] for h in halves])
        y = np.concatenate([labels[tr]] * (1 + len(halves)))
        mu, sd = X.mean(0), X.std(0) + 1e-12
        hits += _lda_fit_predict((X - mu) / sd, y, ((M[k] - mu) / sd)[None])[0] == labels[k]
    return hits / n
'''

SOURCE_FILES['scripts/02_preprocess_features.py'] = r'''"""Step 2 - preprocess every recording and extract unit-invariant EEG descriptors."""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.data import read_any, preprocess
from neuronai.features import extract_features, feature_matrix, FEATURE_NAMES, psd, normalised_log_psd

ap = argparse.ArgumentParser()
ap.add_argument('--asd', default='data/ASD')
ap.add_argument('--td', default='data/TD')
ap.add_argument('--out', default='results')
ap.add_argument('--tag', default='')
ap.add_argument('--block', type=int, default=19,
                help='recordings per child in sequential file numbering (Duville et al.: 1 rest + 18 task = 19); use 1 if one file per child')
a = ap.parse_args()
rows, win_rows, psds, psds1, psds2, topo, failed = [], [], {}, {}, {}, {}, []
for grp, folder in [('ASD', a.asd), ('TD', a.td)]:
    files = sorted(glob.glob(os.path.join(folder, '**', '*.*'), recursive=True))
    files = [f for f in files if os.path.splitext(f)[1].lower() in ('.gdf', '.edf', '.set', '.bdf', '.mat')]
    files.sort(key=lambda p: int((re.findall(r'(\d+)\.\w+$', p) or ['0'])[0]))
    for p in files:
        sid = os.path.splitext(os.path.basename(p))[0]
        try:
            x, fs, names = read_any(p)
            pp = preprocess(x, fs, names)
            W = np.stack(pp['windows'])                  # (n_win, 17, 5000)
        except Exception as exc:                         # unreadable or too short: log and continue
            failed.append({'id': sid, 'group': grp, 'file': os.path.relpath(p, folder), 'error': repr(exc)[:300]})
            print(f"{sid:10s} SKIPPED ({exc!r})", flush=True)
            continue
        F = feature_matrix(extract_features(W, pp['fs']))  # (n_win, 10)
        f, P = psd(W, pp['fs'])
        fsel, lp = normalised_log_psd(f, P.mean(0))
        psds[sid] = lp
        band = (f >= 2) & (f <= 30)                      # spatial-spectral fingerprint (used to verify child blocks)
        Pc = P.mean(0)[:, band]
        topo[sid] = np.log10(Pc / Pc.sum() + 1e-20).ravel()
        half = len(W) // 2
        psds1[sid] = normalised_log_psd(f, P[:half].mean(0))[1]
        psds2[sid] = normalised_log_psd(f, P[half:].mean(0))[1]
        num = int((re.findall(r'(\d+)$', sid) or ['1'])[0])
        k_child = (num - 1) // a.block
        child = f"{grp}-{chr(ord('A') + k_child) if k_child < 26 else f'{k_child + 1:02d}'}"
        cond = 'rest' if a.block > 1 and (num - 1) % a.block == 0 else ('task' if a.block > 1 else 'rest')
        rec = {'id': sid, 'group': grp, 'child': child, 'condition': cond, 'fs_orig': fs, 'duration_s': x.shape[1] / fs,
               'clean_s': pp['clean_sec'], 'n_windows': len(W), 'bad_channels': ','.join(pp['bad_channels']),
               'n_bad': len(pp['bad_channels']), 'line_peaks_hz': ';'.join(f'{v:.2f}' for v in pp['line_peaks']), 'medcorr': pp['medcorr'], 'excluded': pp['excluded'],
               'reason': pp['reason']}
        rec.update({k: v for k, v in zip(FEATURE_NAMES, F.mean(0))})
        rec.update({k + '_h1': v for k, v in zip(FEATURE_NAMES, F[:half].mean(0))})
        rec.update({k + '_h2': v for k, v in zip(FEATURE_NAMES, F[half:].mean(0))})
        rows.append(rec)
        for wi in range(len(W)):
            win_rows.append({'id': sid, 'group': grp, 'child': child, 'window': wi, **dict(zip(FEATURE_NAMES, F[wi]))})
        print(f"{sid:10s} win={len(W)} bad={pp['bad_channels']} excl={pp['excluded']} {pp['reason']}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, f'empirical_features{a.tag}.csv'), index=False)
pd.DataFrame(win_rows).to_csv(os.path.join(a.out, f'empirical_features_windows{a.tag}.csv'), index=False)
np.savez_compressed(os.path.join(a.out, f'empirical_psd{a.tag}.npz'), freqs=fsel, ids=np.array(list(psds)),
                    logpsd=np.stack(list(psds.values())), logpsd_h1=np.stack(list(psds1.values())),
                    logpsd_h2=np.stack(list(psds2.values())), topo=np.stack(list(topo.values())))
if failed:
    pd.DataFrame(failed).to_csv(os.path.join(a.out, f'failed_files{a.tag}.csv'), index=False)
    print(len(failed), 'files could not be read; see failed_files.csv')
print(df.groupby(['group', 'excluded']).size())
'''


# ----------------------------------------------------------------------------- setup
import os, sys, re, json, glob, time, shutil, zipfile, subprocess
import requests
REPO = "/content/NeuronAI"
for rel, src in SOURCE_FILES.items():
    path = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
OUT = "/content/neuronai_full_cohort"
DATA = "/content/data"
os.makedirs(OUT, exist_ok=True)
EEG_EXT = (".gdf", ".edf", ".bdf", ".set", ".fdt", ".mat")
ARCHIVE_EXT = (".zip",)
API = "https://data.mendeley.com/public-api/datasets"
HEAD = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*"}


def as_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("results", "files", "items", "data", "folders"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return []


def file_entry(it, folder=""):
    if not isinstance(it, dict):
        return None
    cd = it.get("content_details") or {}
    url = cd.get("download_url") or it.get("download_url")
    if not url:
        return None
    size = int(cd.get("size", 0) or it.get("size", 0) or 0)
    return (folder, it.get("filename") or it.get("name") or "file", size, url)


def list_dataset(ds, ver):
    """Return [(folder_path, filename, size, url)] using the public Mendeley Data API."""
    # 1) dataset record: https://data.mendeley.com/public-api/datasets/<id>  (works without a token)
    for params in ({"version": ver}, {}):
        try:
            r = requests.get(f"{API}/{ds}", params=params, headers=HEAD, timeout=60)
            print(f"  GET {r.url} -> {r.status_code}")
            if r.ok:
                rec = r.json()
                items = [e for e in (file_entry(it) for it in as_list(rec.get("files", rec))) if e]
                if items:
                    return items
                print("  record keys:", list(rec)[:20] if isinstance(rec, dict) else type(rec))
            else:
                print("  ", r.text[:200].replace("\n", " "))
        except Exception as exc:
            print("  dataset record failed:", exc)
    # 2) folder-by-folder listing
    folders = []
    try:
        r = requests.get(f"{API}/{ds}/folders/{ver}", headers=HEAD, timeout=60)
        if r.ok:
            folders = [f for f in as_list(r.json()) if isinstance(f, dict) and "id" in f]
    except Exception as exc:
        print("  folder listing failed:", exc)
    by_id = {f["id"]: f for f in folders}

    def path_of(fid):
        parts = []
        while fid in by_id:
            parts.append(by_id[fid].get("name", ""))
            fid = by_id[fid].get("parent_id")
        return "/".join(reversed(parts))

    out = []
    for fid in ["root"] + list(by_id):
        try:
            r = requests.get(f"{API}/{ds}/files", params={"folder_id": fid, "version": ver}, headers=HEAD, timeout=60)
            print(f"  GET {r.url} -> {r.status_code}")
        except Exception as exc:
            print("  file listing failed:", exc); continue
        if not r.ok:
            continue
        for it in as_list(r.json()):
            url = (it.get("content_details") or {}).get("download_url") or it.get("download_url")
            if url:
                out.append(("" if fid == "root" else path_of(fid), it.get("filename", "file"), int(it.get("size", 0) or 0), url))
    return out


def download(url, dest):
    tmp = dest + ".part"
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=300, allow_redirects=True, headers={"User-Agent": HEAD["User-Agent"]}) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            os.replace(tmp, dest)
            return True
        except Exception as exc:
            print(f"    attempt {attempt + 1}: {exc}")
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code in (401, 403, 404):
                break
            time.sleep(5 * (attempt + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def extract_archives(folder):
    for zp in glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True):
        try:
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(os.path.splitext(zp)[0])
            os.remove(zp)
        except Exception as exc:
            print("  could not extract", zp, exc)


def short_name(rel, maxlen=100):
    """Short, unique file name for a relative path (keeps the extension and the last folder)."""
    import hashlib
    parts = rel.replace("\\", "/").split("/")
    base = "__".join(parts[-2:]) if len(parts) > 1 else parts[0]
    stem, ext = os.path.splitext(base)
    h = hashlib.md5(rel.encode()).hexdigest()[:6]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[: maxlen - len(ext) - 8]
    return f"{stem}_{h}{ext}"


def prefix_names(folder, grp):
    """Make recording ids unique across groups and folders (EEGLAB .set/.fdt pairs are left untouched)."""
    eeg = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
           if f.lower().endswith((".gdf", ".edf", ".bdf", ".mat"))]
    counts = {}
    for f in eeg:
        counts[os.path.basename(f).lower()] = counts.get(os.path.basename(f).lower(), 0) + 1
    renamed = []
    for f in eeg:
        b = os.path.basename(f)
        new = b
        if counts[b.lower()] > 1:                     # same name in several folders: add a folder tag
            tag = re.sub(r"[^A-Za-z0-9]+", "", os.path.basename(os.path.dirname(f)))[:20] or "x"
            new = f"{tag}_{new}"
        if not new.upper().startswith(grp):
            new = f"{grp}_{new}"
        if new != b:
            os.rename(f, os.path.join(os.path.dirname(f), new))
        renamed.append({"group": grp, "original_path": os.path.relpath(f, folder), "recording_id": os.path.splitext(new)[0]})
    return renamed


manifest = []
for grp, (ds, ver) in DATASETS.items():
    dest = os.path.join(DATA, grp)
    os.makedirs(dest, exist_ok=True)
    print(f"\n=== {grp}: Mendeley dataset {ds} v{ver}")
    have = [f for f in glob.glob(os.path.join(dest, "**", "*"), recursive=True) if f.lower().endswith(EEG_EXT[:4])]
    if have:
        print(f"  {len(have)} EEG files already in this Colab session; download skipped")
        continue
    if DIRECT_URLS.get(grp, "").strip():
        zp = os.path.join(DATA, f"{grp}_all.zip")
        print("  downloading the archive from the link provided ...", flush=True)
        t0 = time.time()
        if not download(DIRECT_URLS[grp].strip(), zp):
            raise RuntimeError(f"The {grp} link did not work (it may have expired). Copy a fresh link (see the "
                               "instructions at the top) and run the cell again.")
        print(f"  {os.path.getsize(zp) / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min; extracting ...", flush=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dest)
        os.remove(zp)
        extract_archives(dest)
        manifest.append({"group": grp, "folder": "", "original_name": "Download All archive", "local_name": "",
                         "size": 0})
        continue
    if grp in MANUAL_ZIPS:
        with zipfile.ZipFile(MANUAL_ZIPS[grp]) as zf:
            zf.extractall(dest)
        extract_archives(dest)
        continue
    items = list_dataset(ds, ver)
    if not items:
        raise RuntimeError(f"Mendeley blocked the automatic download of {grp} (browser check). Follow the "
                           "instructions at the top of the cell: copy the 'Download All' link from your browser "
                           "into DIRECT_URLS and run the cell again.")
    total = sum(s for *_, s, _ in items)
    print(f"  {len(items)} files, {total / 1e9:.2f} GB")
    t0 = time.time()
    for k, (fpath, name, size, url) in enumerate(items, 1):
        ext = os.path.splitext(name)[1].lower()
        small_meta = ext not in EEG_EXT + ARCHIVE_EXT and size < 50e6
        if ext not in EEG_EXT + ARCHIVE_EXT and not small_meta:
            continue
        local = name if name.upper().startswith(grp) else f"{grp}_{name}"
        sub = os.path.join(dest, fpath) if ext in ARCHIVE_EXT else (dest if ext in EEG_EXT else os.path.join(OUT, "metadata", grp))
        os.makedirs(sub, exist_ok=True)
        target = os.path.join(sub, local if ext in EEG_EXT else name)
        if not (os.path.exists(target) and (size == 0 or os.path.getsize(target) == size)):
            download(url, target)
        manifest.append({"group": grp, "folder": fpath, "original_name": name, "local_name": os.path.basename(target),
                         "size": size})
        if k % 25 == 0:
            print(f"  {k}/{len(items)} files  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    extract_archives(dest)

import pandas as pd
inventory, id_map = [], []
for grp in DATASETS:
    root = os.path.join(DATA, grp)
    for f in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not os.path.isfile(f):
            continue
        ext = os.path.splitext(f)[1].lower()
        inventory.append({"group": grp, "original_path": os.path.relpath(f, root), "size": os.path.getsize(f)})
        if ext in (".xlsx", ".xls", ".csv", ".tsv", ".txt", ".json", ".docx", ".pdf", ".md") and os.path.getsize(f) < 50e6:
            mdir = os.path.join(OUT, "metadata", grp)
            os.makedirs(mdir, exist_ok=True)
            shutil.copy(f, os.path.join(mdir, short_name(os.path.relpath(f, root))))
    id_map.extend(prefix_names(root, grp))
pd.DataFrame(id_map).to_csv(os.path.join(OUT, "recording_id_map.csv"), index=False)
pd.DataFrame(inventory).to_csv(os.path.join(OUT, "file_inventory.csv"), index=False)
pd.DataFrame(manifest).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
for grp in DATASETS:
    n = len([p for p in glob.glob(os.path.join(DATA, grp, "**", "*"), recursive=True) if p.lower().endswith(EEG_EXT[:4])])
    print(grp, "EEG files available:", n)
    if n == 0:
        raise RuntimeError(f"No EEG files found for {grp}; see the messages above.")

# ----------------------------------------------------------------------------- preprocessing (scripts/02)
cmd = (f"python scripts/02_preprocess_features.py --asd {DATA}/ASD --td {DATA}/TD --out {OUT} --block {BLOCK}")
print("\n$", cmd, flush=True)
p = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
log = []
for line in p.stdout:
    log.append(line)
    print(line, end="", flush=True)
p.wait()
open(os.path.join(OUT, "preprocess_log.txt"), "w").write("".join(log))
if p.returncode != 0:
    raise RuntimeError("Preprocessing failed; see the log above.")

# ----------------------------------------------------------------------------- package & download
zip_path = shutil.make_archive("/content/NeuronAI_full_cohort_features", "zip", OUT)
print("\nResult:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.1f} MB)")
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs("/content/drive/MyDrive/NeuronAI", exist_ok=True)
    shutil.copy(zip_path, "/content/drive/MyDrive/NeuronAI/")
    print("Copied to MyDrive/NeuronAI/")
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    pass
