# =============================================================================
# NeuronAI v2.0 - Google Colab version (complete pipeline)
# Biophysical EEG digital twins with simulation-based inference for
# personalised in silico neuropharmacology.
#
# HOW TO USE
#   1. Open https://colab.research.google.com and create a new notebook.
#   2. Get the two "Download All" links of the Duville et al. datasets:
#      open https://data.mendeley.com/datasets/7zf99hxxc9/1 (ASD), click "Download All",
#      open your browser's downloads page (Chrome: chrome://downloads), right-click the file,
#      "Copy link address", paste it in DIRECT_URLS["ASD"] below (you can cancel the local download);
#      repeat with https://data.mendeley.com/datasets/spwnt8t25y/1 (TD).
#      Alternative: put both zip files in Google Drive and use MANUAL_ZIPS.
#   3. Paste this whole file into ONE code cell and press Run (Ctrl+Enter). Keep the tab open.
#   4. At the end, the results and figures are downloaded as NeuronAI_results.zip.
#
# MODE = "quick": 2,048 simulations per bank (checks the pipeline; results are less precise).
# MODE = "paper": 8,192 simulations per bank (reproduces the article; about 8-10 h on 2 CPUs).
# =============================================================================
MODE = "quick"            # "quick", "paper" or "test"
DIRECT_URLS = {"ASD": "", "TD": ""}
MANUAL_ZIPS = {}          # e.g. {"ASD": "/content/drive/MyDrive/ASD.zip", "TD": "/content/drive/MyDrive/TD.zip"}
RUN_TESTS = True          # run the unit tests first

# ----------------------------------------------------------------------------- source code (plain text)
SOURCE_FILES = {}
SOURCE_FILES['scripts/00_prepare_data.py'] = r'''"""Step 0 - prepare the complete Duville et al. datasets (as downloaded from Mendeley Data) for NeuronAI.

Usage:  python scripts/00_prepare_data.py --asd <ASD download folder or zip> --td <TD download folder or zip>

The Mendeley archives contain EEG/PartNN/<recording>.gdf plus spreadsheets (Demographics, ASRS, Behaviour).
TD file names lack a group prefix (Part05_restEO.gdf) and the same name exists in both cohorts, so every
recording is linked into data/<GROUP>/ under a unique id (<GROUP>_PartNN_<condition>), and the mapping is
written to results/recording_id_map.csv. Spreadsheets are copied to data_meta/<GROUP>/ with short names.
Nothing is modified in the original download."""
import argparse, glob, hashlib, os, re, shutil, zipfile
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--asd', required=True)
ap.add_argument('--td', required=True)
ap.add_argument('--data', default='data')
ap.add_argument('--meta', default='data_meta')
ap.add_argument('--results', default='results')
a = ap.parse_args()
EEG = ('.gdf', '.edf', '.bdf', '.set', '.mat')
META = ('.xlsx', '.xls', '.csv', '.tsv', '.txt', '.json')


def short_name(rel, maxlen=100):
    parts = rel.replace('\\', '/').split('/')
    base = '__'.join(parts[-2:]) if len(parts) > 1 else parts[0]
    stem, ext = os.path.splitext(base)
    h = hashlib.md5(rel.encode()).hexdigest()[:6]
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)[: maxlen - len(ext) - 8]}_{h}{ext}"


def source_root(path, grp):
    if os.path.isfile(path) and path.lower().endswith('.zip'):
        dest = os.path.join(a.data, f'_extracted_{grp}')
        if not os.path.isdir(dest):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest)
        for zp in glob.glob(os.path.join(dest, '**', '*.zip'), recursive=True):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(os.path.splitext(zp)[0])
            os.remove(zp)
        return dest
    return path


rows = []
os.makedirs(a.results, exist_ok=True)
for grp, src in [('ASD', a.asd), ('TD', a.td)]:
    root = source_root(src, grp)
    out = os.path.join(a.data, grp); os.makedirs(out, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(f))
    eeg = [f for f in files if f.lower().endswith(EEG)]
    names = [os.path.basename(f).lower() for f in eeg]
    for f in eeg:
        b = os.path.basename(f)
        new = b
        if names.count(b.lower()) > 1:
            new = f"{re.sub(r'[^A-Za-z0-9]+', '', os.path.basename(os.path.dirname(f)))[:20]}_{new}"
        if not new.upper().startswith(grp):
            new = f'{grp}_{new}'
        dst = os.path.join(out, new)
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(f), dst)
            except OSError:
                shutil.copy(f, dst)
        rows.append({'group': grp, 'original_path': os.path.relpath(f, root), 'recording_id': os.path.splitext(new)[0]})
    md = os.path.join(a.meta, grp); os.makedirs(md, exist_ok=True)
    for f in files:
        if f.lower().endswith(META) and os.path.getsize(f) < 50e6:
            shutil.copy(f, os.path.join(md, short_name(os.path.relpath(f, root))))
    print(grp, len(eeg), 'recordings linked into', out)
pd.DataFrame(rows).to_csv(os.path.join(a.results, 'recording_id_map.csv'), index=False)
print('mapping written to', os.path.join(a.results, 'recording_id_map.csv'))
'''

SOURCE_FILES['scripts/02c_assign_participants.py'] = r'''"""Step 2c - assign every recording to its participant and condition, and collect participant metadata.

The complete Duville et al. datasets are organised as EEG/PartNN/<file>.gdf with file names that encode
the condition (restEO = 2-min eyes-open rest; scenarioK_IAPS / _4discs / _8discs = task blocks under low,
intermediate and high attentional load). This script replaces the provisional child labels written by
step 2 with the true participant identifiers and adds age, sex, ASRS and tracking-capacity data.

Inputs : results/empirical_features.csv, results/empirical_features_windows.csv,
         results/recording_id_map.csv (written by the Colab download cell), --meta folder with the
         dataset spreadsheets (Age_Gender, PartNN_ASRS, Tracking_Capacity) in ASD/ and TD/ sub-folders.
Outputs: updated empirical_features*.csv and results/participants.csv
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results')
ap.add_argument('--meta', default='data_meta')
a = ap.parse_args()
R = a.results

# 'Idiscs' (one participant) is the image-viewing (IAPS) block and 'bolas' is Spanish for discs; both are
# naming variants in the original files (verified: each participant has 1 rest + 6 blocks of each load).
LOAD = {'restEO': 'rest', 'IAPS': 'low', 'Idiscs': 'low', '4discs': 'intermediate', '4bolas': 'intermediate',
        '8discs': 'high', '8bolas': 'high'}

m = pd.read_csv(f'{R}/recording_id_map.csv')
m['part'] = m.original_path.str.extract(r'Part(\d+)', expand=False).astype(int)
m['cond_raw'] = m.recording_id.str.extract(r'(restEO|IAPS|Idiscs|4discs|4bolas|8discs|8bolas)$', expand=False)
m['scenario'] = m.recording_id.str.extract(r'scenario(\d)', expand=False).fillna(0).astype(int)
chk = m.assign(c=m.cond_raw.map(LOAD)).groupby(['group', 'part']).c.value_counts().unstack(fill_value=0)
assert (chk[['rest', 'low', 'intermediate', 'high']].values == [1, 6, 6, 6]).all(), chk
if m.part.isna().any() or m.cond_raw.isna().any():
    sys.exit('Some recordings could not be mapped to a participant/condition')
m['child'] = m.group + '-P' + m.part.map('{:02d}'.format)
m['condition'] = m.cond_raw.map(LOAD)
lab = m.set_index('recording_id')[['child', 'condition', 'scenario', 'part']]

for fn in ['empirical_features.csv', 'empirical_features_windows.csv']:
    df = pd.read_csv(f'{R}/{fn}')
    df['child'] = lab.loc[df.id, 'child'].values
    if 'condition' in df:
        df['condition'] = lab.loc[df.id, 'condition'].values
        df['scenario'] = lab.loc[df.id, 'scenario'].values
    df.to_csv(f'{R}/{fn}', index=False)

# ------------------------------------------------------------------ participant metadata
def first(pattern):
    fs = sorted(glob.glob(pattern), key=len)
    return fs[0] if fs else None


rows = []
for grp in ['ASD', 'TD']:
    ag = first(os.path.join(a.meta, grp, '*Age_Gender*.xlsx'))
    tc = first(os.path.join(a.meta, grp, '*Tracking_Capacity*.xlsx'))
    age = pd.read_excel(ag) if ag else None
    trk = pd.read_excel(tc) if tc else None
    for p in range(1, 32):
        rec = {'group': grp, 'child': f'{grp}-P{p:02d}', 'participant': p}
        if age is not None:
            r = age[age.iloc[:, 0] == p]
            if len(r):
                rec['age'] = float(r.iloc[0, 1]); rec['sex'] = str(r.iloc[0, 2]).strip()
        if trk is not None:
            r = trk[trk.iloc[:, 0] == p]
            if len(r):
                rec['tracking_capacity'] = float(r.iloc[0, 1])
        f = first(os.path.join(a.meta, grp, f'*Part{p:02d}_ASRS*.xlsx'))
        if f:
            s = pd.read_excel(f)
            for _, row in s.iterrows():
                name = str(row.iloc[0])
                code = re.search(r'\(([A-Z]+)\s*-', name)
                key = 'Total' if name.startswith('Total') else ('DSM5' if name.startswith('DSM') else (code.group(1) if code else None))
                if key:
                    rec[f'ASRS_{key}_T'] = float(row.iloc[2])
                    rec[f'ASRS_{key}_raw'] = float(row.iloc[1])
        rows.append(rec)
pp = pd.DataFrame(rows)
pp.to_csv(f'{R}/participants.csv', index=False)
q = pd.read_csv(f'{R}/empirical_features.csv')
print(q.groupby(['group', 'condition', 'excluded']).size().unstack(fill_value=0))
print(pp.groupby('group')[['age', 'ASRS_Total_T']].describe().round(1).T)
print(pp.groupby(['group', 'sex']).size())
'''

SOURCE_FILES['scripts/fig6_clinical_load.py'] = r'''"""Figure 6 - clinical association (ASRS) and attentional-load effects of twin parameters (complete cohort)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fig_style import setup, save, panel, C_TD, C_ASD, C_SIM, C_TXT, C_TXT2, GRID, DIV_NEG, DIV_MID, DIV_POS
from neuronai.model import FIT_PARAMS, FIT_LABELS
from neuronai.features import FEATURE_NAMES, FEATURE_LABELS

R = 'results'
if not (os.path.exists(f'{R}/asrs_association.csv') and os.path.exists(f'{R}/load_effects.csv')):
    sys.exit('Figure 9 needs asrs_association.csv and load_effects.csv (complete cohort only)')
setup()
asrs = pd.read_csv(f'{R}/asrs_association.csv')
ld = pd.read_csv(f'{R}/load_effects.csv').set_index('variable')
LAB = {**FIT_LABELS, **FEATURE_LABELS}
VARS = FIT_PARAMS + FEATURE_NAMES
cm = LinearSegmentedColormap.from_list('div', [DIV_NEG, DIV_MID, DIV_POS])

fig = plt.figure(figsize=(7.2, 6.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.05], height_ratios=[1, 1], hspace=0.55, wspace=0.95)

# (a) ASRS total: Spearman rho within each cohort
ax = fig.add_subplot(gs[:, 0]); panel(ax, 'a', x=-0.9)
cols = [('ASD', 'ASD'), ('TD', 'TD'), ('all', 'all')]
M = np.full((len(VARS), len(cols)), np.nan); Q = np.ones_like(M)
for j, (c, _) in enumerate(cols):
    t = asrs[(asrs.cohort == c) & (asrs.score == 'ASRS_Total_T')].set_index('variable')
    for i, v in enumerate(VARS):
        if v in t.index:
            M[i, j] = t.loc[v, 'spearman_rho']; Q[i, j] = t.loc[v, 'q_fdr']
im = ax.imshow(M, cmap=cm, vmin=-0.6, vmax=0.6, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:+.2f}' + ('*' if Q[i, j] < 0.05 else ''), ha='center', va='center', fontsize=5.3,
                color='white' if abs(M[i, j]) > 0.4 else C_TXT, fontweight='bold' if Q[i, j] < 0.05 else 'normal')
ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols], fontsize=6.3)
ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top'); ax.tick_params(axis='x', length=0, pad=2)
ax.set_yticks(range(len(VARS))); ax.set_yticklabels([LAB[v] for v in VARS], fontsize=5.8)
ax.axhline(len(FIT_PARAMS) - 0.5, color=C_TXT, lw=0.8)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, orientation='horizontal', location='bottom')
cb.set_label('Spearman ρ with ASRS total T (child means; * q < 0.05)', fontsize=5.6); cb.ax.tick_params(labelsize=5.5)
ax.set_title('Autistic-trait severity', pad=16)

# (b) attentional load: effect sizes (dz) of task vs rest and high vs low load
ax = fig.add_subplot(gs[:, 1]); panel(ax, 'b', x=-0.45)
yy = np.arange(len(VARS))[::-1]
for off, col, q, lab, c in [(0.18, 'task_minus_rest_dz', 'task_vs_rest_wilcoxon_q', 'task − rest', C_SIM),
                            (-0.18, 'high_minus_low_dz', 'friedman_load_q', 'high − low load', '#6250d6')]:
    v = ld.loc[VARS, col].values; qq = ld.loc[VARS, q].values
    ax.barh(yy + off, v, height=0.34, color=[c if x < 0.05 else '#d9d8d3' for x in qq], label=lab)
ax.axvline(0, color='#9a9994', lw=0.7)
ax.set_yticks(yy); ax.set_yticklabels([LAB[v] for v in VARS], fontsize=5.8)
ax.axhline(len(FEATURE_NAMES) - 0.5, color=C_TXT, lw=0.8)
ax.set_xlabel("within-child effect size d$_z$", fontsize=6.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_SIM, label='task − rest'), Patch(color='#6250d6', label='high − low load'),
                   Patch(color='#d9d8d3', label='q ≥ 0.05')], fontsize=5.6, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3,
          handlelength=1.0, columnspacing=0.8)
ax.set_title(f"Attentional load ({int(ld['n_children_load'].iloc[0])} children)", pad=8, fontsize=7.5)

# (c, d) strongest twin-parameter associations with ASRS within each cohort
pp = pd.read_csv(f'{R}/participants.csv').set_index('child')
tw = pd.read_csv(f'{R}/twin_parameters.csv')
for k, (g, colr) in enumerate([('ASD', C_ASD), ('TD', C_TD)]):
    ax = fig.add_subplot(gs[k, 2]); panel(ax, 'cd'[k], x=-0.32)
    t = asrs[(asrs.cohort == g) & (asrs.score == 'ASRS_Total_T') & (asrs.kind == 'parameter')].sort_values('p')
    v = t.iloc[0]
    cmn = tw.groupby('child')[f"{v.variable}_full"].mean()
    cmn = cmn[[c for c in cmn.index if c.startswith(g)]]
    y = pp.loc[cmn.index, 'ASRS_Total_T']
    ax.scatter(cmn.values, y.values, s=12, color=colr, lw=0)
    b1, b0 = np.polyfit(cmn.values, y.values, 1)
    xs = np.linspace(cmn.min(), cmn.max(), 10); ax.plot(xs, b0 + b1 * xs, color=C_TXT2, lw=0.8, ls='--')
    ax.set_xlabel(f'{FIT_LABELS[v.variable]} (child mean)', fontsize=6.3); ax.set_ylabel('ASRS total T', fontsize=6.3)
    ax.text(0.03, 0.97, f"ρ = {v.spearman_rho:+.2f}\np = {v.p:.3f}, q = {v.q_fdr:.2f}", transform=ax.transAxes, va='top', fontsize=5.8, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    ax.set_title(f"{g}: strongest parameter", fontsize=7)
save(fig, 'Figure6_clinical_load')
print('figure 9 done')
'''

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
        try:
            self.P = np.linalg.inv(lw.covariance_)
        except np.linalg.LinAlgError:                  # degenerate reference (very few recordings)
            self.P = np.linalg.pinv(lw.covariance_)

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

SOURCE_FILES['scripts/fig_style.py'] = r'''"""Shared figure style for NeuronAI publication figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt

C_TD = '#2a78d6'      # blue
C_ASD = '#eb6834'     # orange
C_SIM = '#1baf7a'     # aqua (model / simulation)
C_NULL = '#8a8984'    # neutral gray
C_TXT = '#0b0b0b'
C_TXT2 = '#52514e'
GRID = '#e6e5e1'
DIV_NEG, DIV_MID, DIV_POS = '#2a78d6', '#f0efec', '#e34948'
CHILD_ASD = ['#eb6834', '#c2491b', '#f39a6f', '#8f3410']
CHILD_TD = ['#2a78d6', '#184f95', '#6da7ec', '#0d366b']


def setup():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.edgecolor': '#9a9994', 'axes.linewidth': 0.6, 'axes.labelcolor': C_TXT,
        'xtick.color': C_TXT2, 'ytick.color': C_TXT2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'legend.frameon': False, 'savefig.dpi': 600, 'figure.dpi': 150,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'lines.linewidth': 1.4, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left', color=C_TXT)


def save(fig, name, outdir='figures'):
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/{name}.png')
    fig.savefig(f'{outdir}/{name}.pdf')
    fig.savefig(f'{outdir}/{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
'''

SOURCE_FILES['scripts/01_generate_banks.py'] = r'''"""Step 1 - generate the simulation banks (inverse bank I and emulator bank E)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuronai.bank import generate_bank

ap = argparse.ArgumentParser()
ap.add_argument('--kind', default='I', choices=['I', 'E'])
ap.add_argument('--n', type=int, default=8192)
ap.add_argument('--seed', type=int, default=None)
ap.add_argument('--out', default='results')
a = ap.parse_args()
seed = a.seed if a.seed is not None else (2026 if a.kind == 'I' else 3031)
os.makedirs(a.out, exist_ok=True)
generate_bank(a.n, a.kind, seed=seed, out_path=os.path.join(a.out, f'bank_{a.kind}.npz'))
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

SOURCE_FILES['scripts/03_train_models.py'] = r'''"""Step 3 - train the amortised inverse ensemble and the forward emulator; evaluate
parameter recovery (identifiability), conformal coverage and emulator accuracy on
held-out simulations."""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.bank import load_bank
from neuronai.model import FIT_PARAMS, FIT_BOUNDS, params_to_unit
from neuronai.inference import InverseEnsemble, ForwardEmulator, SpectralEmulator, EMU_INPUTS, emu_matrix
from neuronai.features import FEATURE_NAMES

R = 'results'
bI, bE = load_bank(f'{R}/bank_I.npz'), load_bank(f'{R}/bank_E.npz')


def theta_dict(b):
    return {n: b['theta'][:, i] for i, n in enumerate(b['theta_names'])}


# ---------------- inverse model (bank I) ----------------
ok = bI['ok']
X = bI['X'][ok]
Z = params_to_unit(theta_dict(bI))[ok]
rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
n_tr, n_cal = int(0.7 * len(X)), int(0.1 * len(X))
tr, cal, te = perm[:n_tr], perm[n_tr:n_tr + n_cal], perm[n_tr + n_cal:]
inv = InverseEnsemble().fit(X[tr], Z[tr], X[cal], Z[cal], coverage=0.90)
pred = inv.predict(X[te])
lo, hi = inv.interval(X[te])
rows = []
for i, n in enumerate(FIT_PARAMS):
    r2 = 1 - ((pred[:, i] - Z[te, i]) ** 2).sum() / ((Z[te, i] - Z[te, i].mean()) ** 2).sum()
    cov = float(((Z[te, i] >= lo[:, i]) & (Z[te, i] <= hi[:, i])).mean())
    width = float((hi[:, i] - lo[:, i]).mean())
    rows.append({'parameter': n, 'R2': r2, 'pearson_r': np.corrcoef(pred[:, i], Z[te, i])[0, 1],
                 'conformal_coverage_90': cov, 'mean_interval_width_unit': width,
                 'contraction': 1 - width})
rec = pd.DataFrame(rows)
rec.to_csv(f'{R}/recovery_inverse.csv', index=False)
np.savez_compressed(f'{R}/recovery_test.npz', pred=pred, true=Z[te], lo=lo, hi=hi)
print(rec.round(3))

# ---------------- forward emulator (banks I + E) ----------------
tE = theta_dict(bE)
tI = theta_dict(bI)
UE, UI = emu_matrix(tE), emu_matrix(tI)
Xe = np.concatenate([bI['X'][bI['ok']], bE['X'][bE['ok']]])
Ue = np.concatenate([UI[bI['ok']], UE[bE['ok']]])
perm = np.random.default_rng(1).permutation(len(Xe))
n_tr = int(0.85 * len(Xe))
emu = ForwardEmulator().fit(Ue[perm[:n_tr]], Xe[perm[:n_tr]])
sc = emu.score(Ue[perm[n_tr:]], Xe[perm[n_tr:]], FEATURE_NAMES)
pd.DataFrame({'feature': list(sc), 'R2_heldout': list(sc.values())}).to_csv(f'{R}/emulator_accuracy.csv', index=False)
print('Emulator held-out R2:', {k: round(v, 3) for k, v in sc.items()})

# ---------------- spectral emulator (full 1-30 Hz relative log-spectrum) ----------------
LPe = np.concatenate([bI['logpsd'][bI['ok']], bE['logpsd'][bE['ok']]])
semu = SpectralEmulator(seed=11).fit(Ue[perm[:n_tr]], LPe[perm[:n_tr]])
Ph = semu.predict(Ue[perm[n_tr:]]); Tt = LPe[perm[n_tr:]]
spec_r2 = 1 - ((Ph - Tt) ** 2).sum(1) / ((Tt - Tt.mean(1, keepdims=True)) ** 2).sum(1)
print('Spectral emulator: median held-out spectral R2 =', round(float(np.median(spec_r2)), 3))
pd.DataFrame({'heldout_spectral_R2': spec_r2}).to_csv(f'{R}/spectral_emulator_accuracy.csv', index=False)

with open(f'{R}/models.pkl', 'wb') as f:
    pickle.dump({'inverse': inv, 'emulator': emu, 'spec_emulator': semu}, f)
json.dump({'n_bank_I': int(ok.sum()), 'n_bank_E': int(bE['ok'].sum()), 'n_train_inv': len(tr),
           'n_cal_inv': len(cal), 'n_test_inv': len(te), 'conformal_q': inv.q.tolist()},
          open(f'{R}/training_summary.json', 'w'), indent=2)
'''

SOURCE_FILES['scripts/03b_noise_ceiling.py'] = r'''"""Step 3b - intrinsic stochasticity of 20-s simulations: fraction of descriptor variance
explained by parameters (noise ceiling for any emulator / inverse model)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.bank import sample_unit
from neuronai.model import params_from_unit, simulate, FIT_PARAMS
from neuronai.features import extract_features, feature_matrix, FEATURE_NAMES

N_P, N_S = 128, 4
U = sample_unit(N_P, len(FIT_PARAMS), seed=99)
th = params_from_unit(np.repeat(U, N_S, axis=0))
F = []
for s in range(0, N_P * N_S, 128):
    r = simulate({k: v[s:s + 128] for k, v in th.items()}, duration=20.0, seed=123 + s)
    F.append(feature_matrix(extract_features(r.eeg, r.fs)))
F = np.concatenate(F).reshape(N_P, N_S, -1)
within = F.var(axis=1, ddof=1).mean(0)
total = F.reshape(-1, F.shape[-1]).var(0, ddof=1)
ceil = 1 - within / total
pd.DataFrame({'feature': FEATURE_NAMES, 'noise_ceiling_R2': ceil,
              'within_param_sd': np.sqrt(within)}).to_csv('results/noise_ceiling.csv', index=False)
print(dict(zip(FEATURE_NAMES, np.round(ceil, 3))))
'''

SOURCE_FILES['scripts/03c_benchmark_inference.py'] = r'''"""Step 3c - comparison of inverse-inference strategies on held-out simulations.

Methods compared (identical training bank and held-out test set):
  1. kNN-ABC: mean parameters of the 20 nearest simulations in standardised descriptor space
  2. Random forest regression (300 trees)
  3. NeuronAI amortised deep ensemble (posterior mean)
  4. NeuronAI amortised + emulator-gradient refinement within conformal intervals (proposed)
Metrics: parameter recovery (R^2), data-space discrepancy of the estimate (descriptors and full
log-spectrum re-simulated with the stochastic simulator) and computing time per recording."""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from neuronai.bank import load_bank
from neuronai.model import FIT_PARAMS, params_to_unit, params_from_unit, simulate
from neuronai.features import extract_features, feature_matrix, psd, normalised_log_psd
from neuronai.inference import refine_twin

R = 'results'
N_TEST_SIM = 150
b = load_bank(f'{R}/bank_I.npz')
ok = b['ok']
X = b['X'][ok]; LP = b['logpsd'][ok]
Z = params_to_unit({n: b['theta'][:, i] for i, n in enumerate(b['theta_names'])})[ok]
rng = np.random.default_rng(0)                      # identical split to step 3
perm = rng.permutation(len(X))
n_tr, n_cal = int(0.7 * len(X)), int(0.1 * len(X))
tr, te = perm[:n_tr], perm[n_tr + n_cal:]
te = te[:N_TEST_SIM]
m = pickle.load(open(f'{R}/models.pkl', 'rb')); inv, emu, semu = m['inverse'], m['emulator'], m['spec_emulator']
sc = StandardScaler().fit(X[tr])
scale = X[tr].std(0)
est, times = {}, {}
t = time.time()
Xs_tr = sc.transform(X[tr])
knn = []
for x in sc.transform(X[te]):
    d = ((Xs_tr - x) ** 2).sum(1)
    knn.append(Z[tr][np.argsort(d)[:20]].mean(0))
est['kNN-ABC'] = np.array(knn); times['kNN-ABC'] = (time.time() - t) / len(te)
t = time.time()
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=0).fit(X[tr], Z[tr])
t_fit = time.time() - t
t = time.time(); est['Random forest'] = rf.predict(X[te]); times['Random forest'] = (time.time() - t) / len(te)
t = time.time(); est['NeuronAI amortised'] = inv.predict(X[te]); times['NeuronAI amortised'] = (time.time() - t) / len(te)
t = time.time()
lo, hi = inv.interval(X[te])
ref = []
for k in range(len(te)):
    nn = np.argsort((((X[te][k] - X[tr]) / scale) ** 2).sum(1))[:5]
    z, _ = refine_twin(X[te][k], emu, np.vstack([est['NeuronAI amortised'][k], Z[tr][nn]]), scale, lo[k], hi[k],
                       spectrum=LP[te][k], spec_emulator=semu, spec_scale=LP[tr].std(0))
    ref.append(z)
est['NeuronAI amortised + refinement'] = np.array(ref)
times['NeuronAI amortised + refinement'] = (time.time() - t) / len(te) + times['NeuronAI amortised']

est['True parameters (noise ceiling)'] = Z[te]; times['True parameters (noise ceiling)'] = float('nan')
rows = []
for name, E in est.items():
    r2 = 1 - ((E - Z[te]) ** 2).sum(0) / ((Z[te] - Z[te].mean(0)) ** 2).sum(0)
    th = params_from_unit(E)
    res = simulate(th, duration=20.0, seed=777)       # same noise seed for every method
    F = feature_matrix(extract_features(res.eeg, res.fs))
    f, P = psd(res.eeg, res.fs); _, lp = normalised_log_psd(f, P)
    E_lp = LP[te]
    spec_r2 = 1 - ((E_lp - lp) ** 2).sum(1) / ((E_lp - E_lp.mean(1, keepdims=True)) ** 2).sum(1)
    feat_err = np.abs((F - X[te]) / scale).mean(1)
    rows.append({'method': name, 'mean_R2_parameters': float(r2.mean()),
                 'R2_identifiable_(B,rho,K,A)': float(r2[[FIT_PARAMS.index(p) for p in ['B', 'rho', 'K', 'A']]].mean()),
                 'median_spectral_R2_resimulated': float(np.median(spec_r2)),
                 'median_descriptor_error_SD': float(np.median(feat_err)),
                 'seconds_per_recording': times[name]})
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv(f'{R}/inference_benchmark.csv', index=False)
print(out.round(3).to_string())
'''

SOURCE_FILES['scripts/04_fit_subjects.py'] = r'''"""Step 4 - fit a digital twin to every recording (full data and split halves) and run
posterior-predictive and out-of-sample validation against the real EEG.

Recordings are processed in fixed blocks of 21 (21 x 6 seeds = 126 simulations per batch), and every block
is simulated with a seed derived from its position, so results are identical whether the step is run in one
process or split into shards (--shard k --nshards n; the last shard to finish merges the outputs)."""
import argparse, glob, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.model import FIT_PARAMS, params_from_unit, params_to_unit, simulate
from neuronai.inference import refine_twin
from neuronai.bank import load_bank
from neuronai.features import FEATURE_NAMES, extract_features, feature_matrix, psd, normalised_log_psd

ap = argparse.ArgumentParser()
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ARGS = ap.parse_args()
R = 'results'
N_SEEDS = 6
BLOCK = 21
models = pickle.load(open(f'{R}/models.pkl', 'rb'))
inv, emu, semu = models['inverse'], models['emulator'], models['spec_emulator']
df = pd.read_csv(f'{R}/empirical_features.csv')
df = df[~df.excluded].reset_index(drop=True)
df_all = df
blocks = np.arange(len(df)) // BLOCK
mine = (blocks % ARGS.nshards) == ARGS.shard
TAG = '' if ARGS.nshards == 1 else f'_shard{ARGS.shard}of{ARGS.nshards}'
# scale of meaningful variation: between-recording SD of each descriptor in the TD cohort (all recordings)
SCALE = df_all[df_all.group == 'TD'][FEATURE_NAMES].std().values
bank = load_bank(f'{R}/bank_I.npz')
BX = bank['X'][bank['ok']]
BZ = params_to_unit({n: bank['theta'][:, i] for i, n in enumerate(bank['theta_names'])})[bank['ok']]
emp = np.load(f'{R}/empirical_psd.npz')
eid = list(emp['ids'])
SPEC = {t: np.stack([emp[k][eid.index(s_)] for s_ in df_all.id]) for t, k in
        [('full', 'logpsd'), ('h1', 'logpsd_h1'), ('h2', 'logpsd_h2')]}
SPEC_SCALE = SPEC['full'][(df_all.group == 'TD').values].std(0)
SPEC = {t: v[mine] for t, v in SPEC.items()}
BLK = blocks[mine]
df = df_all[mine].reset_index(drop=True)
out = df[['id', 'group', 'child', 'condition']].copy()
units = {}
for tag, cols in [('full', FEATURE_NAMES), ('h1', [f + '_h1' for f in FEATURE_NAMES]),
                  ('h2', [f + '_h2' for f in FEATURE_NAMES])]:
    Xs = df[cols].values
    mu = inv.predict(Xs)                      # amortised posterior mean (deep ensemble)
    lo, hi = inv.interval(Xs)                 # 90 % split-conformal intervals
    Z, L = [], []
    for k in range(len(Xs)):
        nn = np.argsort((((Xs[k] - BX) / SCALE) ** 2).sum(1))[:5]   # nearest simulations as extra starts
        z, loss = refine_twin(Xs[k], emu, np.vstack([mu[k], BZ[nn]]), SCALE, lo[k], hi[k],
                              spectrum=SPEC[tag][k], spec_emulator=semu, spec_scale=SPEC_SCALE)
        Z.append(z); L.append(loss)
    Z = np.array(Z)
    th, tlo, thi, tnn = params_from_unit(Z), params_from_unit(lo), params_from_unit(hi), params_from_unit(mu)
    for n in FIT_PARAMS:
        out[f'{n}_{tag}'] = th[n]
        if tag == 'full':
            out[f'{n}_lo'], out[f'{n}_hi'], out[f'{n}_amortised'] = tlo[n], thi[n], tnn[n]
    out[f'refine_loss_{tag}'] = L
    units[tag] = Z
    print(f'twins ({tag}) refined; median standardised loss {np.median(L):.2f}', flush=True)
out.to_csv(f'{R}/twin_parameters{TAG}.csv', index=False)


def ppc(unit, seed0):
    """Simulate each twin with N_SEEDS independent noise realisations (20 s each), block by block."""
    th_all = params_from_unit(np.repeat(unit, N_SEEDS, axis=0))
    feats, lps = [], []
    for b in np.unique(BLK):
        rows = np.flatnonzero(np.repeat(BLK == b, N_SEEDS))
        res = simulate({k: v[rows] for k, v in th_all.items()}, duration=20.0, seed=seed0 + int(b))
        feats.append(feature_matrix(extract_features(res.eeg, res.fs)))
        f, P = psd(res.eeg, res.fs)
        fsel, lp = normalised_log_psd(f, P)
        lps.append(lp)
    n = len(unit)
    F = np.concatenate(feats).reshape(n, N_SEEDS, -1)
    LP = np.concatenate(lps).reshape(n, N_SEEDS, -1)
    return F, LP, fsel


F_full, LP_full, fsel = ppc(units['full'], 50000)
F_h1, _, _ = ppc(units['h1'], 60000)
F_amort, LP_amort, _ = ppc(inv.predict(df[FEATURE_NAMES].values), 70000)   # ablation: no refinement
np.savez_compressed(f'{R}/ppc{TAG}.npz', F_full=F_full, LP_full=LP_full, F_h1=F_h1, F_amort=F_amort, LP_amort=LP_amort, freqs=fsel,
                    ids=df['id'].values)
print('PPC simulations done:', F_full.shape, flush=True)

# ---------------------------------------------------------------- merge shards
if ARGS.nshards > 1:
    parts = [f'{R}/twin_parameters_shard{k}of{ARGS.nshards}.csv' for k in range(ARGS.nshards)]
    pz = [f'{R}/ppc_shard{k}of{ARGS.nshards}.npz' for k in range(ARGS.nshards)]
    if all(os.path.exists(p) for p in parts + pz):
        order = list(df_all.id)
        tw = pd.concat([pd.read_csv(p) for p in parts]).set_index('id').loc[order].reset_index()
        tw.to_csv(f'{R}/twin_parameters.csv', index=False)
        Z = [np.load(p, allow_pickle=True) for p in pz]
        ids = list(np.concatenate([z['ids'] for z in Z])); pos = [ids.index(i) for i in order]
        merged = {k: np.concatenate([z[k] for z in Z])[pos] for k in ['F_full', 'LP_full', 'F_h1', 'F_amort', 'LP_amort']}
        np.savez_compressed(f'{R}/ppc.npz', **merged, freqs=Z[0]['freqs'], ids=np.array(order))
        print('merged', len(tw), 'twins', flush=True)
'''

SOURCE_FILES['scripts/05_pharmaco_validity.py'] = r'''"""Step 5 - pharmacological face validity: simulate canonical drug classes on every
fitted twin and compare the predicted direction of EEG change with the direction
reported in the human pharmaco-EEG literature (benchmarks fixed a priori)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from neuronai.model import CONTROLS, FIT_PARAMS, simulate
from neuronai.features import FEATURE_NAMES, extract_features, feature_matrix
from neuronai.stats import cluster_bootstrap_ci, sign_flip_test

R = 'results'
N_SEEDS = 4
tw = pd.read_csv(f'{R}/twin_parameters.csv')
theta = {n: tw[f'{n}_full'].values for n in FIT_PARAMS}
n = len(tw)


def u_of(**kw):
    u = np.zeros(len(CONTROLS))
    for k, v in kw.items():
        u[CONTROLS.index(k)] = v
    return u


DRUGS = [
    # name, control vector, feature, expected sign, literature
    ('GABA_A positive modulator (benzodiazepine class)', u_of(GABA_A=1.0), 'rel_beta', +1,
     'Feshchenko et al., 1997; Mandema & Danhof, 1992'),
    ('NMDA receptor antagonist (ketamine class)', u_of(NMDA_2A=-1.0, NMDA_2B=-1.0), 'rel_alpha', -1,
     'Muthukumaraswamy et al., 2015; de la Salle et al., 2016'),
    ('5-HT2A agonist (psilocybin class)', u_of(**{'5HT': 1.0}), 'rel_alpha', -1,
     'Muthukumaraswamy et al., 2013; Kometer et al., 2015'),
    ('DA/NE enhancer (stimulant class)', u_of(DA=1.0, NE=1.0), 'theta_beta', -1,
     'Clarke et al., 2017; Loo et al., 2004'),
    ('Slow-inhibition agonist (baclofen class)', u_of(GABA_slow=1.0), 'rel_delta', +1,
     'exploratory (no firm human benchmark)'),
]


def run(u_vec, seed):
    rows = np.repeat(np.arange(n), N_SEEDS)
    th = {k: v[rows] for k, v in theta.items()}
    U = np.repeat(u_vec[None, :], len(rows), axis=0)
    feats = []
    for s in range(0, len(rows), 128):
        sl = slice(s, s + 128)
        res = simulate({k: v[sl] for k, v in th.items()}, u=U[sl], duration=20.0, seed=seed + s)
        feats.append(feature_matrix(extract_features(res.eeg, res.fs)))
    return np.concatenate(feats).reshape(n, N_SEEDS, -1)


base = run(np.zeros(len(CONTROLS)), 777)    # common random numbers across conditions
records, effects = [], {}
for name, u, feat, sign, ref in DRUGS:
    drug = run(u, 777)
    d = (drug - base).mean(1)                   # paired, per twin
    eff = pd.DataFrame(d, columns=FEATURE_NAMES)
    eff['theta_beta'] = (drug[..., FEATURE_NAMES.index('rel_theta')] - drug[..., FEATURE_NAMES.index('rel_beta')]).mean(1) - \
        (base[..., FEATURE_NAMES.index('rel_theta')] - base[..., FEATURE_NAMES.index('rel_beta')]).mean(1)
    eff['id'] = tw['id'].values; eff['group'] = tw['group'].values; eff['child'] = tw['child'].values; eff['drug'] = name
    effects[name] = eff
    v = eff[feat].values
    m, lo, hi = cluster_bootstrap_ci(v, tw['child'].values)
    p = wilcoxon(v).pvalue
    child_means = pd.Series(v).groupby(tw['child'].values).mean()
    child_agree = float((np.sign(child_means) == sign).mean())
    p_child = sign_flip_test(child_means.values)[1] if len(child_means) >= 6 else np.nan
    agree = float((np.sign(v) == sign).mean())
    records.append({'drug': name, 'feature': feat, 'expected_sign': '+' if sign > 0 else '-',
                    'mean_change': m, 'ci_lo': lo, 'ci_hi': hi, 'wilcoxon_p_records': p,
                    'pct_twins_expected_direction': 100 * agree, 'pct_children_expected_direction': 100 * child_agree,
                    'p_child_signflip': p_child, 'n_children': int(len(child_means)),
                    'concordant': bool(np.sign(m) == sign and (lo > 0 if sign > 0 else hi < 0)), 'reference': ref})
    print(records[-1], flush=True)
pd.DataFrame(records).to_csv(f'{R}/pharmaco_validity.csv', index=False)
pd.concat(effects.values()).to_csv(f'{R}/pharmaco_effects_all.csv', index=False)
'''

SOURCE_FILES['scripts/05b_control_fingerprints.py'] = r'''"""Step 5b - mechanistic fingerprints of the pharmacological controls.

For every fitted twin, each control is set to +1 (maximal agonism/potentiation)
in isolation and the change of every descriptor is measured with common random
numbers. In addition, post hoc mechanistically refined drug models (clearly
separated from the pre-specified benchmarks of step 5) are evaluated."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.model import CONTROLS, FIT_PARAMS, simulate
from neuronai.features import FEATURE_NAMES, extract_features, feature_matrix
from neuronai.stats import cluster_bootstrap_ci

R = 'results'
N_SEEDS = 2
tw = pd.read_csv(f'{R}/twin_parameters.csv')
n = len(tw)
rows = np.repeat(np.arange(n), N_SEEDS)
theta = {k: tw[f'{k}_full'].values[rows] for k in FIT_PARAMS}


def run(u=None, extra=None, seed=31):
    th = {k: v.copy() for k, v in theta.items()}
    if extra:
        for k, f in extra.items():
            th[k] = th.get(k, np.ones(len(rows))) * f
    F = []
    for s in range(0, len(rows), 103):
        sl = slice(s, s + 103)
        nb = min(103, len(rows) - s)
        uu = None if u is None else np.repeat(np.asarray(u, float)[None], nb, 0)
        r = simulate({k: v[sl] for k, v in th.items()}, u=uu, duration=20.0, seed=seed + s)
        F.append(feature_matrix(extract_features(r.eeg, r.fs)))
    F = np.concatenate(F).reshape(n, N_SEEDS, -1)
    tb = F[..., FEATURE_NAMES.index('rel_theta')] - F[..., FEATURE_NAMES.index('rel_beta')]
    return np.concatenate([F, tb[..., None]], -1).mean(1)


names = FEATURE_NAMES + ['theta_beta']
base = run()
out = []
for m, c in enumerate(CONTROLS):
    u = np.zeros(len(CONTROLS)); u[m] = 1.0
    d = run(u) - base
    for j, f in enumerate(names):
        mu, lo, hi = cluster_bootstrap_ci(d[:, j], tw.child.values, n_boot=2000)
        out.append({'control': c, 'feature': f, 'mean': mu, 'ci_lo': lo, 'ci_hi': hi,
                    'sd': d[:, j].std(ddof=1), 'pct_positive': 100 * (d[:, j] > 0).mean()})
    print('control', c, 'done', flush=True)
pd.DataFrame(out).to_csv(f'{R}/control_fingerprints.csv', index=False)

POSTHOC = [
    ('GABA_A modulator at both inhibitory synapses + prolonged IPSC decay (post hoc)', dict(G=1.3, B=1.3, g_tau=1.3), 'exponent', +1,
     'steeper aperiodic exponent with GABAergic potentiation (Gao et al., 2017)'),
    ('GABA_A modulator at both inhibitory synapses + prolonged IPSC decay (post hoc)', dict(G=1.3, B=1.3, g_tau=1.3), 'rel_beta', +1,
     'beta increase with benzodiazepines (Feshchenko et al., 1997)'),
    ('NMDA antagonist, interneuron-preferential + AMPA increase (post hoc)', dict(NI_scale=0.3, N=0.7, A=1.15), 'rel_alpha', -1,
     'alpha decrease with ketamine (Muthukumaraswamy et al., 2015)'),
]
ph = []
for name, extra, feat, sign, ref in POSTHOC:
    d = run(extra=extra) - base
    j = names.index(feat)
    mu, lo, hi = cluster_bootstrap_ci(d[:, j], tw.child.values, n_boot=2000)
    ph.append({'model': name, 'feature': feat, 'expected_sign': '+' if sign > 0 else '-', 'mean_change': mu,
               'ci_lo': lo, 'ci_hi': hi, 'pct_expected': 100 * (np.sign(d[:, j]) == sign).mean(),
               'concordant': bool(np.sign(mu) == sign and (lo > 0 if sign > 0 else hi < 0)), 'benchmark': ref})
    print(ph[-1], flush=True)
pd.DataFrame(ph).to_csv(f'{R}/pharmaco_posthoc.csv', index=False)
'''

SOURCE_FILES['scripts/06_intervention.py'] = r'''"""Step 6 - personalised in silico pharmacological search for every recording.

Design ("emulator proposes, simulator disposes"):
 1. Emulator search over all 129 sparse supports (grid + L-BFGS-B refinement).
 2. Screening: the 12 best emulator candidates AND 12 random control vectors matched one-to-one in
    sparsity and norm are simulated with the full stochastic simulator (8 paired noise streams).
    The best candidate of each arm is selected on the screening streams.
 3. Confirmation: the two winners and the baseline are re-simulated with 12 FRESH paired streams,
    giving selection-unbiased improvements (no winner's curse) and bootstrap CIs.
 4. Robustness of the emulator ranking (ensemble members, descriptor subsets) and TD leave-one-child-out
    negative control.

Normative reference: TD recordings of the SAME condition (rest, or the same attentional load), excluding the
child under study (leave-one-child-out for TD recordings). When participant/condition labels are available
(complete cohort), the search is run on a pre-specified subset: for every child, the resting-state recording
and the first valid block of each attentional load (at most four recordings per child); use --all to run
every recording.
"""
import argparse, glob, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.model import CONTROLS, FIT_PARAMS
from neuronai.features import FEATURE_NAMES
from neuronai.intervention import (NormativeReference, search, grid_controls, grid_rank, member_deltas,
                                   random_like, simulate_candidates, verified_improvement)

ap = argparse.ArgumentParser()
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--all', action='store_true')
ARGS = ap.parse_args()
R = 'results'
N_CAND, N_SCREEN, N_CONFIRM = 12, 8, 12
MIN_REF = 10          # fewer condition-matched TD recordings than this: use all TD recordings of other children
models = pickle.load(open(f'{R}/models.pkl', 'rb'))
emu = models['emulator']
df = pd.read_csv(f'{R}/empirical_features.csv')
df = df[~df.excluded].reset_index(drop=True)
tw = pd.read_csv(f'{R}/twin_parameters.csv').set_index('id')
X = df[FEATURE_NAMES].values
is_td = (df.group == 'TD').values
COND = df['condition'].astype(str).values
MATCH = len(set(COND) - {'rest', 'task'}) > 0          # true load labels available (complete cohort)
if MATCH and not ARGS.all:
    first = (df.assign(_o=df.get('scenario', 0)).sort_values(['child', 'condition', '_o'])
             .groupby(['child', 'condition']).head(1).index)
    SELECT = np.zeros(len(df), bool); SELECT[first] = True
else:
    SELECT = np.ones(len(df), bool)
print('recordings selected for the intervention search:', SELECT.sum(), flush=True)
SPECTRAL = [FEATURE_NAMES.index(f) for f in ['rel_delta', 'rel_theta', 'rel_alpha', 'rel_beta', 'iaf', 'exponent']]
SYNC = [FEATURE_NAMES.index(f) for f in ['plv_alpha', 'wpli_alpha', 'kuramoto_alpha']]
GRID = grid_controls()
rng_boot = np.random.default_rng(2026)


def boot_ci(x_s, delta, ref, n=2000):
    d0 = ref.distance(x_s[None])[0]
    k = delta.shape[0]
    vals = [100 * (d0 - ref.distance((x_s + delta[rng_boot.integers(0, k, k)].mean(0))[None])[0]) / d0 for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


rows, top = [], {}
for i, r in df.iterrows():
    if not SELECT[i] or int(np.flatnonzero(SELECT).tolist().index(i)) % ARGS.nshards != ARGS.shard:
        continue
    sid = r['id']
    ref_mask = is_td & (df['child'].values != r['child'])     # TD: leave-one-child-out reference
    if MATCH and (ref_mask & (COND == COND[i])).sum() >= MIN_REF:
        ref_mask &= COND == COND[i]                           # condition-matched normative reference
    ref = NormativeReference(X[ref_mask])
    theta = {n: tw.loc[sid, f'{n}_full'] for n in FIT_PARAMS}
    supports, U_all, own = GRID
    md = member_deltas(theta, emu, U_all)
    ens = md.mean(0)
    res = search(X[i], theta, emu, ref, n_refine=N_CAND, precomputed=(supports, U_all, own, ens))
    cands = res['results'][:N_CAND]
    U_ai = np.array([c['u'] for c in cands])
    U_rn = random_like(U_ai, seed=1000 + i)
    # ---- screening (paired streams) ----
    U_scr = np.vstack([np.zeros((1, len(CONTROLS))), U_ai, U_rn])
    F = simulate_candidates(theta, U_scr, N_SCREEN, seed=10_000 + i)
    _, m_ai, _ = verified_improvement(X[i], F[1:1 + N_CAND], F[0], ref)
    _, m_rn, _ = verified_improvement(X[i], F[1 + N_CAND:], F[0], ref)
    k_ai, k_rn = int(np.argmax(m_ai)), int(np.argmax(m_rn))
    # ---- confirmation (fresh paired streams) ----
    Fc = simulate_candidates(theta, np.vstack([np.zeros((1, len(CONTROLS))), U_ai[k_ai], U_rn[k_rn]]),
                             N_CONFIRM, seed=20_000 + i)
    per_ai, mean_ai, d_ai = verified_improvement(X[i], Fc[1:2], Fc[0], ref)
    per_rn, mean_rn, d_rn = verified_improvement(X[i], Fc[2:3], Fc[0], ref)
    lo_ai, hi_ai = boot_ci(X[i], d_ai[0], ref)
    lo_rn, hi_rn = boot_ci(X[i], d_rn[0], ref)
    win = cands[k_ai]
    # ---- robustness of the emulator ranking ----
    member_best = [grid_rank(X[i], md[m], U_all, own, supports, ref) for m in range(len(md))]
    top1 = res['results'][0]
    jacc = [len(set(mb['support']) & set(top1['support'])) / len(set(mb['support']) | set(top1['support']))
            for mb in member_best]
    sub = {tag: grid_rank(X[i], ens, U_all, own, supports, NormativeReference(X[ref_mask], idx))['name']
           for tag, idx in [('spectral', SPECTRAL), ('sync', SYNC)]}
    rank_corr = float(np.corrcoef(np.array([c['improvement'] for c in cands]), m_ai)[0, 1])
    rec = {'id': sid, 'group': r['group'], 'child': r['child'], 'condition': r['condition'], 'D0': res['d0'],
           'best_combo': win['name'], 'n_targets': len(win['support']), 'emulator_rank_of_selected': k_ai + 1,
           'improvement_emulated': win['improvement'],
           'improvement_screen': float(m_ai[k_ai]),
           'improvement_confirmed': float(mean_ai[0]), 'ci_lo': lo_ai, 'ci_hi': hi_ai,
           'pct_streams_improved': float((per_ai[0] > 0).mean() * 100),
           'random_best_combo': '+'.join(CONTROLS[j] for j in np.where(np.abs(U_rn[k_rn]) > 1e-9)[0]),
           'random_improvement_confirmed': float(mean_rn[0]), 'random_ci_lo': lo_rn, 'random_ci_hi': hi_rn,
           'screen_median_ai': float(np.median(m_ai)), 'screen_median_random': float(np.median(m_rn)),
           'screen_pct_ai_positive': float((m_ai > 0).mean() * 100),
           'screen_pct_random_positive': float((m_rn > 0).mean() * 100),
           'emulator_vs_simulator_rank_r': rank_corr,
           'emulator_top1_combo': top1['name'],
           'member_agreement_top1': float(np.mean([mb['name'] == top1['name'] for mb in member_best])),
           'member_target_jaccard': float(np.mean(jacc)),
           'best_spectral_only': sub['spectral'], 'best_sync_only': sub['sync']}
    rec.update({f'u_{c}': win['u'][j] for j, c in enumerate(CONTROLS)})
    rec.update({f'delta_{f}': d_ai[0].mean(0)[j] for j, f in enumerate(FEATURE_NAMES)})
    rows.append(rec)
    top[sid] = {'emulator_candidates': [{'combo': c['name'], 'u': c['u'].tolist(), 'emulated': c['improvement'],
                                         'screen': float(s)} for c, s in zip(cands, m_ai)],
                'random_screen': m_rn.tolist()}
    print(time.strftime("%H:%M:%S"), f"{sid:10s} {r['group']} D0={res['d0']:.2f} sel={win['name']:26s} emu={win['improvement']:5.1f}% "
          f"confirmed={mean_ai[0]:6.1f}% [{lo_ai:6.1f},{hi_ai:6.1f}]  random={mean_rn[0]:6.1f}%", flush=True)

out = pd.DataFrame(rows)
tag = '' if ARGS.nshards == 1 else f'_shard{ARGS.shard}of{ARGS.nshards}'
out.to_csv(f'{R}/intervention_results{tag}.csv', index=False)
json.dump(top, open(f'{R}/intervention_candidates{tag}.json', 'w'))
if ARGS.nshards > 1:
    parts = sorted(glob.glob(f'{R}/intervention_results_shard*of{ARGS.nshards}.csv'))
    if len(parts) == ARGS.nshards:
        allr = pd.concat([pd.read_csv(p) for p in parts])
        allr = allr.set_index('id').loc[[s for s in df.id if s in set(allr.id)]].reset_index()
        allr.to_csv(f'{R}/intervention_results.csv', index=False)
        cand = {}
        for p in sorted(glob.glob(f'{R}/intervention_candidates_shard*of{ARGS.nshards}.json')):
            cand.update(json.load(open(p)))
        json.dump(cand, open(f'{R}/intervention_candidates.json', 'w'))
        print('merged', len(allr), 'recordings')
        out = allr
print(out.groupby('group')[['D0', 'improvement_emulated', 'improvement_confirmed', 'random_improvement_confirmed']].median())
'''

SOURCE_FILES['scripts/07_validation_stats.py'] = r'''"""Step 7 - validation statistics against real EEG and child-level inference.

Outputs (results/):
  ppc_summary.csv            posterior-predictive and out-of-sample agreement per feature
  ppc_spectral_r2.csv        per-recording R^2 between simulated and empirical log-spectra
  reliability.csv            split-half ICC(2,1), between-child ICC(1) of twin parameters
  identification.json        child identification from twin parameters (leave-one-recording-out)
  group_params.csv           ASD vs TD twin parameters (child-level permutation, cluster bootstrap)
  intervention_summary.json  aggregate intervention statistics, nulls and consistency
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.model import FIT_PARAMS, CONTROLS
from neuronai.features import FEATURE_NAMES
from neuronai.stats import (icc_ci, icc_1, cluster_bootstrap_ci, child_permutation_test, hedges_g, fdr_bh,
                            identify_discriminant, sign_flip_test)
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare

R = 'results'
df = pd.read_csv(f'{R}/empirical_features.csv'); df = df[~df.excluded].reset_index(drop=True)
tw = pd.read_csv(f'{R}/twin_parameters.csv')
ppc = np.load(f'{R}/ppc.npz', allow_pickle=True)
emp_psd = np.load(f'{R}/empirical_psd.npz')
child = df['child'].values
summary = {}

# ------------------------------------------------------------- 1. PPC & out-of-sample
F_sim = ppc['F_full'].mean(1)       # in-sample twin prediction
F_oos = ppc['F_h1'].mean(1)         # twin fitted on first half -> predicts second half
X = df[FEATURE_NAMES].values
X1 = df[[f + '_h1' for f in FEATURE_NAMES]].values
X2 = df[[f + '_h2' for f in FEATURE_NAMES]].values
rows = []
for j, f in enumerate(FEATURE_NAMES):
    # leave-one-child-out group-mean baseline (population model without personalisation)
    base = np.array([X2[(df.group == g) & (df.child != c), j].mean() for g, c in zip(df.group, df.child)])
    mae_tw = np.abs(F_oos[:, j] - X2[:, j]).mean()
    mae_base = np.abs(base - X2[:, j]).mean()
    mae_rt = np.abs(X1[:, j] - X2[:, j]).mean()
    # within-child (condition-to-condition) tracking: correlation after removing child means
    dm = lambda v: v - pd.Series(v).groupby(child).transform('mean').values
    rows.append({'feature': f,
                 'r_in_sample': np.corrcoef(F_sim[:, j], X[:, j])[0, 1],
                 'r_out_of_sample': np.corrcoef(F_oos[:, j], X2[:, j])[0, 1],
                 'r_within_child_oos': np.corrcoef(dm(F_oos[:, j]), dm(X2[:, j]))[0, 1],
                 'r_test_retest': np.corrcoef(X1[:, j], X2[:, j])[0, 1],
                 'MAE_twin_oos': mae_tw, 'MAE_group_mean': mae_base, 'MAE_test_retest': mae_rt,
                 'skill_vs_group_mean': 1 - mae_tw / mae_base,
                 'bias_in_sample': float((F_sim[:, j] - X[:, j]).mean())})
ppc_tab = pd.DataFrame(rows)
ppc_tab.to_csv(f'{R}/ppc_summary.csv', index=False)
print(ppc_tab.round(3).to_string())

ids = list(emp_psd['ids'])
E = np.stack([emp_psd['logpsd'][ids.index(s)] for s in df.id])
S = ppc['LP_full'].mean(1)
r2 = 1 - ((E - S) ** 2).sum(1) / ((E - E.mean(1, keepdims=True)) ** 2).sum(1)
# spectral R^2 of the leave-one-child-out group-mean spectrum (non-personalised baseline)
Gm = np.stack([E[(df.group == g).values & (df.child != c).values].mean(0) for g, c in zip(df.group, df.child)])
r2_base = 1 - ((E - Gm) ** 2).sum(1) / ((E - E.mean(1, keepdims=True)) ** 2).sum(1)
pd.DataFrame({'id': df.id, 'group': df.group, 'child': child, 'R2_twin': r2, 'R2_group_mean': r2_base}).to_csv(
    f'{R}/ppc_spectral_r2.csv', index=False)
summary['spectral_R2_twin'] = cluster_bootstrap_ci(r2, child, np.median)
summary['spectral_R2_group_mean'] = cluster_bootstrap_ci(r2_base, child, np.median)
summary['spectral_R2_twin_better_pct'] = float((r2 > r2_base).mean() * 100)
# ablation: amortised posterior mean without emulator refinement
Sa = ppc['LP_amort'].mean(1)
r2a = 1 - ((E - Sa) ** 2).sum(1) / ((E - E.mean(1, keepdims=True)) ** 2).sum(1)
summary['spectral_R2_amortised_only'] = cluster_bootstrap_ci(r2a, child, np.median)
Fa = ppc['F_amort'].mean(1)
SC = df[df.group == 'TD'][FEATURE_NAMES].std().values
summary['feature_error_twin_TDsd'] = dict(zip(FEATURE_NAMES, np.round(np.abs(F_sim - X).mean(0) / SC, 3).tolist()))
summary['feature_error_amortised_TDsd'] = dict(zip(FEATURE_NAMES, np.round(np.abs(Fa - X).mean(0) / SC, 3).tolist()))
pd.DataFrame({'id': df.id, 'R2_amortised': r2a}).to_csv(f'{R}/ppc_spectral_r2_amortised.csv', index=False)

# ------------------------------------------------------------- 2. reliability & identifiability of children
rel = []
for n in FIT_PARAMS:
    icc, lo, hi = icc_ci(tw[f'{n}_h1'].values, tw[f'{n}_h2'].values)
    rel.append({'parameter': n, 'ICC_split_half': icc, 'ci_lo': lo, 'ci_hi': hi,
                'ICC1_between_child': icc_1(tw[f'{n}_full'].values, child)})
for f in FEATURE_NAMES:
    icc, lo, hi = icc_ci(df[f + '_h1'].values, df[f + '_h2'].values)
    rel.append({'parameter': 'feature:' + f, 'ICC_split_half': icc, 'ci_lo': lo, 'ci_hi': hi,
                'ICC1_between_child': icc_1(df[f].values, child)})
pd.DataFrame(rel).to_csv(f'{R}/reliability.csv', index=False)

P = tw[[f'{n}_full' for n in FIT_PARAMS]].values
Pz = (P - P.mean(0)) / P.std(0)


def identify(M, labels):
    hits = 0
    for k in range(len(M)):
        m = np.arange(len(M)) != k
        cents = {c: M[m & (labels == c)].mean(0) for c in np.unique(labels[m])}
        pred = min(cents, key=lambda c: np.linalg.norm(M[k] - cents[c]))
        hits += pred == labels[k]
    return hits / len(M)


acc = identify(Pz, child)
rng = np.random.default_rng(0)
N_PERM_ID = 200 if len(child) < 300 else 50
perm = [identify(Pz, rng.permutation(child)) for _ in range(N_PERM_ID)]
chance = max(pd.Series(child).value_counts(normalize=True))
summary['child_identification'] = {'accuracy': acc, 'perm_p': float((1 + sum(p >= acc for p in perm)) / (1 + len(perm))),
                                   'perm_mean': float(np.mean(perm)), 'majority_chance': float(chance),
                                   'n_children': int(len(np.unique(child))), 'n_recordings': int(len(child))}

# discriminant identification (secondary): shrinkage LDA on the nine twin parameters, trained in each fold
# on the training recordings and on the split-half twins of those recordings
H = [tw[[f'{n}_{h}' for n in FIT_PARAMS]].values for h in ('h1', 'h2')]
acc_l = identify_discriminant(P, child, H)
rng = np.random.default_rng(2)
perm_l = []
for _ in range(N_PERM_ID):
    pc = rng.permutation(len(child))          # permute recordings, keeping each recording's halves together
    perm_l.append(identify_discriminant(P[pc], child, [h[pc] for h in H]))
Fobs = df[FEATURE_NAMES].values
summary['child_identification_discriminant'] = {
    'accuracy': acc_l, 'perm_p': float((1 + sum(p >= acc_l for p in perm_l)) / (1 + len(perm_l))), 'perm_mean': float(np.mean(perm_l)),
    'reference_raw_descriptors_discriminant': identify_discriminant(Fobs, child),
    'reference_raw_descriptors_centroid': identify(((Fobs - Fobs.mean(0)) / Fobs.std(0)), child)}
# identification within each cohort (removes any contribution of the acquisition system)
for g in ['ASD', 'TD']:
    m = (df.group == g).values
    summary['child_identification_discriminant'][f'within_{g}'] = {
        'accuracy_centroid': identify(Pz[m], child[m]),
        'accuracy_discriminant': identify_discriminant(P[m], child[m], [h[m] for h in H]),
        'reference_raw_descriptors_discriminant': identify_discriminant(Fobs[m], child[m]),
        'chance': float(1 / len(np.unique(child[m]))), 'n_children': int(len(np.unique(child[m]))),
        'n_recordings': int(m.sum())}
print('child identification: centroid', round(acc, 3), '| discriminant', round(acc_l, 3),
      '| raw descriptors', round(summary['child_identification_discriminant']['reference_raw_descriptors_discriminant'], 3))

# ------------------------------------------------------------- 3. group comparison of twin parameters
gp = []
for n in FIT_PARAMS + FEATURE_NAMES:
    v = tw[f'{n}_full'].values if n in FIT_PARAMS else df[n].values
    diff, p, nperm = child_permutation_test(v, df.group.values, child)
    a, b = v[df.group == 'ASD'], v[df.group == 'TD']
    ca = child[df.group == 'ASD']; cb = child[df.group == 'TD']
    rng = np.random.default_rng(1)
    la, lb = np.unique(ca), np.unique(cb)
    bs = []
    for _ in range(2000):
        sa = np.concatenate([rng.choice(a[ca == c], (ca == c).sum()) for c in rng.choice(la, len(la))])
        sb = np.concatenate([rng.choice(b[cb == c], (cb == c).sum()) for c in rng.choice(lb, len(lb))])
        bs.append(hedges_g(sa, sb))
    gp.append({'variable': n, 'kind': 'parameter' if n in FIT_PARAMS else 'feature',
               'ASD_mean': a.mean(), 'ASD_sd': a.std(ddof=1), 'TD_mean': b.mean(), 'TD_sd': b.std(ddof=1),
               'hedges_g': hedges_g(a, b), 'g_ci_lo': np.percentile(bs, 2.5), 'g_ci_hi': np.percentile(bs, 97.5),
               'p_child_permutation': p, 'n_permutations': nperm})
gp = pd.DataFrame(gp)
gp['q_fdr'] = fdr_bh(gp['p_child_permutation'].values)
gp.to_csv(f'{R}/group_params.csv', index=False)
print(gp.round(3).to_string())

# ------------------------------------------------------------- 4. intervention statistics
iv = pd.read_csv(f'{R}/intervention_results.csv')
asd, td = iv[iv.group == 'ASD'], iv[iv.group == 'TD']
s = {}
for tag, sub in [('ASD', asd), ('TD', td)]:
    s[f'{tag}_D0'] = cluster_bootstrap_ci(sub.D0.values, sub.child.values)
    s[f'{tag}_improvement_emulated'] = cluster_bootstrap_ci(sub.improvement_emulated.values, sub.child.values)
    s[f'{tag}_improvement_confirmed_mean'] = cluster_bootstrap_ci(sub.improvement_confirmed.values, sub.child.values)
    s[f'{tag}_improvement_confirmed_median'] = cluster_bootstrap_ci(sub.improvement_confirmed.values, sub.child.values, np.median)
    s[f'{tag}_random_confirmed_median'] = cluster_bootstrap_ci(sub.random_improvement_confirmed.values, sub.child.values, np.median)
    diff = (sub.improvement_confirmed - sub.random_improvement_confirmed).values
    s[f'{tag}_ai_minus_random_median'] = cluster_bootstrap_ci(diff, sub.child.values, np.median)
    s[f'{tag}_pct_ai_better_than_random'] = float((diff > 0).mean() * 100)
    s[f'{tag}_pct_confirmed_ci_above_zero'] = float((sub.ci_lo > 0).mean() * 100)
    s[f'{tag}_screen_pct_positive_ai'] = float(sub.screen_pct_ai_positive.mean())
    s[f'{tag}_screen_pct_positive_random'] = float(sub.screen_pct_random_positive.mean())
    s[f'{tag}_child_median_ai_minus_random'] = pd.Series(diff).groupby(sub.child.values).median().round(2).to_dict()
    per_child = pd.Series(diff).groupby(sub.child.values).median()
    per_child_conf = pd.Series(sub.improvement_confirmed.values).groupby(sub.child.values).median()
    s[f'{tag}_n_children'] = int(len(per_child))
    s[f'{tag}_n_recordings'] = int(len(sub))
    s[f'{tag}_children_ai_better_pct'] = float((per_child > 0).mean() * 100)
    if len(per_child) >= 6:
        s[f'{tag}_child_level_signflip'] = sign_flip_test(per_child.values, stat=np.median)
        s[f'{tag}_child_level_wilcoxon_p'] = float(wilcoxon(per_child.values).pvalue)
        s[f'{tag}_child_level_confirmed_wilcoxon_p'] = float(wilcoxon(per_child_conf.values).pvalue)
        s[f'{tag}_children_confirmed_positive_pct'] = float((per_child_conf > 0).mean() * 100)
    for cond in sorted(set(sub.condition)):
        cs = sub[sub.condition == cond]
        dd = (cs.improvement_confirmed - cs.random_improvement_confirmed).values
        s[f'{tag}_by_condition'] = s.get(f'{tag}_by_condition', {})
        s[f'{tag}_by_condition'][cond] = {'n': int(len(cs)), 'confirmed_median': float(cs.improvement_confirmed.median()),
                                          'random_median': float(cs.random_improvement_confirmed.median()),
                                          'ai_minus_random_median': float(np.median(dd)),
                                          'pct_ai_better': float((dd > 0).mean() * 100),
                                          'wilcoxon_p': float(wilcoxon(dd).pvalue) if len(dd) >= 6 else None}
if 'ASD_n_children' in s and s['ASD_n_children'] >= 6:
    dif = (iv.improvement_confirmed - iv.random_improvement_confirmed).values
    s['ASD_vs_TD_guidance_advantage_child_perm'] = child_permutation_test(dif, iv.group.values, iv.child.values)[:2]
s['emulated_vs_confirmed_r'] = float(np.corrcoef(iv.improvement_emulated, iv.improvement_confirmed)[0, 1])
s['emulator_optimism_median'] = float(np.median(iv.improvement_emulated - iv.improvement_confirmed))
s['emulator_rank_vs_simulator_r_median'] = float(np.nanmedian(iv.emulator_vs_simulator_rank_r))
s['selected_emulator_rank_median'] = float(iv.emulator_rank_of_selected.median())
s['member_agreement_top1_mean'] = float(iv.member_agreement_top1.mean())
s['member_target_jaccard_mean'] = float(iv.member_target_jaccard.mean())
s['top_combo_frequency_ASD'] = asd.best_combo.value_counts().head(10).to_dict()
s['n_targets_ASD'] = asd.n_targets.value_counts().to_dict()
# recommendation consistency within child: cosine similarity of u* (within vs between children)
U = asd[[f'u_{c}' for c in CONTROLS]].values
Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
Cs = Un @ Un.T
cc = asd.child.to_numpy(dtype=str)
same = (cc[:, None] == cc[None]) & ~np.eye(len(cc), dtype=bool)
diff_ = cc[:, None] != cc[None]
obs = Cs[same].mean() - Cs[diff_].mean()
rng = np.random.default_rng(0)
null = []
for _ in range(2000):
    pc = rng.permutation(cc)
    sm = (pc[:, None] == pc[None]) & ~np.eye(len(pc), dtype=bool)
    null.append(Cs[sm].mean() - Cs[pc[:, None] != pc[None]].mean())
s['u_cosine_within_child'] = float(Cs[same].mean())
s['u_cosine_between_child'] = float(Cs[diff_].mean())
s['u_cosine_perm_p'] = float((1 + sum(n_ >= obs for n_ in null)) / 2001)
# mean control vector per child
s['mean_u_by_child'] = {c: asd[asd.child == c][[f'u_{k}' for k in CONTROLS]].mean().round(3).to_dict()
                        for c in np.unique(cc)}
s['modal_combo_by_child'] = {c: asd[asd.child == c].best_combo.value_counts().index[0] for c in np.unique(cc)}
s['modal_combo_share_by_child'] = {c: float(asd[asd.child == c].best_combo.value_counts(normalize=True).iloc[0])
                                   for c in np.unique(cc)}
s['robust_spectral_only_agree'] = float((asd.best_spectral_only == asd.emulator_top1_combo).mean())
s['robust_sync_only_agree'] = float((asd.best_sync_only == asd.emulator_top1_combo).mean())
summary['intervention'] = s

# ------------------------------------------------------------- 6. clinical association (ASRS) and attentional load
if os.path.exists(f'{R}/participants.csv'):
    pp = pd.read_csv(f'{R}/participants.csv').set_index('child')
    VARS = FIT_PARAMS + FEATURE_NAMES
    cm_ = pd.DataFrame({n: (tw[f'{n}_full'].values if n in FIT_PARAMS else df[n].values) for n in VARS})
    cm_['child'] = child; cm_['group'] = df.group.values
    cmean = cm_.groupby(['child', 'group']).mean().reset_index().set_index('child').join(pp.drop(columns='group'))
    rows = []
    for g in ['ASD', 'TD', 'all']:
        sub = cmean if g == 'all' else cmean[cmean.group == g]
        for n in VARS:
            for score in ['ASRS_Total_T', 'ASRS_SC_T', 'ASRS_UB_T', 'ASRS_SR_T', 'ASRS_DSM5_T']:
                if score not in sub:
                    continue
                ok = sub[[n, score]].dropna()
                rho, p = spearmanr(ok[n], ok[score])
                rows.append({'cohort': g, 'variable': n, 'kind': 'parameter' if n in FIT_PARAMS else 'feature',
                             'score': score, 'n_children': len(ok), 'spearman_rho': rho, 'p': p})
    asrs = pd.DataFrame(rows)
    for (g, sc), idx in asrs.groupby(['cohort', 'score']).groups.items():
        asrs.loc[idx, 'q_fdr'] = fdr_bh(asrs.loc[idx, 'p'].values)
    asrs.to_csv(f'{R}/asrs_association.csv', index=False)
    summary['asrs'] = {'n_children': int(cmean.ASRS_Total_T.notna().sum()),
                       'min_q_total_within_ASD': float(asrs[(asrs.cohort == 'ASD') & (asrs.score == 'ASRS_Total_T')].q_fdr.min()),
                       'min_q_total_within_TD': float(asrs[(asrs.cohort == 'TD') & (asrs.score == 'ASRS_Total_T')].q_fdr.min()),
                       'min_q_total_all': float(asrs[(asrs.cohort == 'all') & (asrs.score == 'ASRS_Total_T')].q_fdr.min())}
    print(asrs[asrs.score == 'ASRS_Total_T'].sort_values('p').head(12).round(3).to_string())

if set(df.condition) >= {'rest', 'low', 'intermediate', 'high'}:
    rows = []
    for n in FIT_PARAMS + FEATURE_NAMES:
        v = tw[f'{n}_full'].values if n in FIT_PARAMS else df[n].values
        t = pd.DataFrame({'v': v, 'child': child, 'cond': df.condition.values, 'group': df.group.values})
        piv = t.groupby(['child', 'cond']).v.mean().unstack()
        full = piv.dropna(subset=['low', 'intermediate', 'high'])
        task = full[['low', 'intermediate', 'high']].mean(1)
        rt = full.dropna(subset=['rest'])
        rec = {'variable': n, 'kind': 'parameter' if n in FIT_PARAMS else 'feature', 'n_children_load': len(full),
               'n_children_rest': len(rt)}
        rec['friedman_load_p'] = float(friedmanchisquare(full['low'], full['intermediate'], full['high']).pvalue)
        dhl = (full['high'] - full['low']).values
        rec['high_minus_low_median'] = float(np.median(dhl)); rec['high_minus_low_dz'] = float(dhl.mean() / dhl.std(ddof=1))
        dtr = (task.loc[rt.index] - rt['rest']).values
        rec['task_minus_rest_median'] = float(np.median(dtr)); rec['task_minus_rest_dz'] = float(dtr.mean() / dtr.std(ddof=1))
        rec['task_vs_rest_wilcoxon_p'] = float(wilcoxon(dtr).pvalue)
        # does the load effect differ between cohorts? (child-level permutation on high-minus-low)
        gch = t.groupby('child').group.first().loc[full.index].values
        rec['load_effect_group_diff_p'] = child_permutation_test(dhl, gch, full.index.values)[1]
        rows.append(rec)
    ld = pd.DataFrame(rows)
    for col in ['friedman_load_p', 'task_vs_rest_wilcoxon_p', 'load_effect_group_diff_p']:
        ld[col.replace('_p', '_q')] = fdr_bh(ld[col].values)
    ld.to_csv(f'{R}/load_effects.csv', index=False)
    print(ld.round(3).to_string())

# ------------------------------------------------------------- 5. dimensionality of the control space
fp = pd.read_csv(f'{R}/control_fingerprints.csv')
SDtd = df[df.group == 'TD'][FEATURE_NAMES].std()
ctrl = list(dict.fromkeys(fp.control))
Mfp = np.array([[fp[(fp.control == c) & (fp.feature == f)]['mean'].iloc[0] / SDtd[f] for f in FEATURE_NAMES] for c in ctrl])
U_, S_, Vt_ = np.linalg.svd(Mfp, full_matrices=False)
summary['control_space'] = {'variance_explained': (S_ ** 2 / (S_ ** 2).sum()).round(4).tolist(),
                            'loading_controls_pc1': dict(zip(ctrl, (U_[:, 0] * S_[0]).round(3).tolist())),
                            'loading_features_pc1': dict(zip(FEATURE_NAMES, Vt_[0].round(3).tolist()))}
json.dump(summary, open(f'{R}/validation_summary.json', 'w'), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:4000])
'''

SOURCE_FILES['scripts/08_single_case.py'] = r'''"""Step 8 - single-case analysis: one recording of the high-autistic-trait (ASD) cohort versus the TD reference.

The case is fixed a priori as the resting-state recording of participant 1 of the ASD cohort
(ASD_Part01_restEO in the complete datasets; ASD_EEG1 in the 120-file subset). The reference is every
valid TD recording of the same condition (resting state). The 30 best emulator candidates and 30 matched random control vectors are
simulated with 20 paired noise streams; candidates are ranked by simulated improvement; the top-ranked
intervention (and the best random vector) are then confirmed on 20 fresh streams."""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from neuronai.model import CONTROLS, FIT_PARAMS, simulate
from neuronai.features import FEATURE_NAMES, psd, normalised_log_psd
from neuronai.intervention import (NormativeReference, search, random_like, simulate_candidates,
                                   verified_improvement)

R = 'results'

N_CAND, N_STREAMS = 30, 20
models = pickle.load(open(f'{R}/models.pkl', 'rb'))
df = pd.read_csv(f'{R}/empirical_features.csv'); df = df[~df.excluded].reset_index(drop=True)
tw = pd.read_csv(f'{R}/twin_parameters.csv').set_index('id')
X = df[FEATURE_NAMES].values
CASE = 'ASD_Part01_restEO' if (df.id == 'ASD_Part01_restEO').any() else 'ASD_EEG1'
REF_MASK = ((df.group == 'TD') & (df.condition == df.loc[df.id == CASE, 'condition'].iloc[0])).values
if REF_MASK.sum() < 10:                     # too few condition-matched TD recordings: use all TD recordings
    REF_MASK = (df.group == 'TD').values
ref = NormativeReference(X[REF_MASK])
print('case', CASE, '| TD reference recordings:', REF_MASK.sum())
i = int(np.where(df.id == CASE)[0][0])
theta = {n: tw.loc[CASE, f'{n}_full'] for n in FIT_PARAMS}
res = search(X[i], theta, models['emulator'], ref, n_refine=N_CAND)
d0 = res['d0']
cands = res['results'][:N_CAND]
U_ai = np.array([c['u'] for c in cands]); U_rn = random_like(U_ai, seed=77)
F = simulate_candidates(theta, np.vstack([np.zeros((1, len(CONTROLS))), U_ai, U_rn]), N_STREAMS, seed=4242)
per_ai, m_ai, d_ai = verified_improvement(X[i], F[1:1 + N_CAND], F[0], ref)
per_rn, m_rn, _ = verified_improvement(X[i], F[1 + N_CAND:], F[0], ref)
rng = np.random.default_rng(0)
rows = []
for k in range(N_CAND):
    bs = [100 * (d0 - ref.distance((X[i] + d_ai[k][rng.integers(0, N_STREAMS, N_STREAMS)].mean(0))[None])[0]) / d0
          for _ in range(2000)]
    u = cands[k]['u']
    rows.append({'combination': cands[k]['name'],
                 'u': ' / '.join(f"{CONTROLS[j]} {u[j]:+.2f}" for j in np.where(np.abs(u) >= 0.05)[0]),  # |u| < 0.05: negligible
                 'emulator_rank': k + 1, 'improvement_emulated': cands[k]['improvement'],
                 'improvement_simulated': float(m_ai[k]), 'ci_lo': float(np.percentile(bs, 2.5)),
                 'ci_hi': float(np.percentile(bs, 97.5)), 'pct_streams_improved': float((per_ai[k] > 0).mean() * 100),
                 'D_initial': d0, 'D_final': d0 * (1 - m_ai[k] / 100),
                 'key': ' '.join(f'{CONTROLS[j]}{np.round(u[j], 1):+.1f}' for j in np.where(np.abs(u) >= 0.1)[0])})
tab = pd.DataFrame(rows).sort_values('improvement_simulated', ascending=False)
tab = tab.drop_duplicates('key').reset_index(drop=True)        # practically identical control vectors
tab.insert(0, 'rank', np.arange(1, len(tab) + 1))
tab.head(10).to_csv(f'{R}/single_case_top10.csv', index=False)
tab.to_csv(f'{R}/single_case_all_candidates.csv', index=False)
print(tab.head(10).round(2).to_string())

# confirmation of the winner and of the best random vector on fresh streams
k_best = int(np.argmax(m_ai)); k_rn = int(np.argmax(m_rn))
Fc = simulate_candidates(theta, np.vstack([np.zeros((1, len(CONTROLS))), U_ai[k_best], U_rn[k_rn]]), N_STREAMS, seed=9191)
pc_ai, mc_ai, dc_ai = verified_improvement(X[i], Fc[1:2], Fc[0], ref)
pc_rn, mc_rn, dc_rn = verified_improvement(X[i], Fc[2:3], Fc[0], ref)
bs_ai = [100 * (d0 - ref.distance((X[i] + dc_ai[0][rng.integers(0, N_STREAMS, N_STREAMS)].mean(0))[None])[0]) / d0 for _ in range(2000)]
bs_rn = [100 * (d0 - ref.distance((X[i] + dc_rn[0][rng.integers(0, N_STREAMS, N_STREAMS)].mean(0))[None])[0]) / d0 for _ in range(2000)]
# spectra of the confirmed winner for the figure (same fresh streams)
lp = []
for U in (np.zeros(len(CONTROLS)), U_ai[k_best]):
    r = simulate({k: np.full(N_STREAMS, v) for k, v in theta.items()}, u=np.repeat(U[None], N_STREAMS, 0),
                 duration=20.0, seed=9191, crn_index=np.arange(N_STREAMS))
    f, P = psd(r.eeg, r.fs); fsel, l = normalised_log_psd(f, P); lp.append(l)
emp = np.load(f'{R}/empirical_psd.npz'); ids = list(emp['ids'])
np.savez_compressed(f'{R}/single_case.npz', freqs=fsel, lp_base=lp[0], lp_best=lp[1],
                    emp_case=emp['logpsd'][ids.index(CASE)],
                    emp_td=emp['logpsd'][[ids.index(s) for s in df.id[REF_MASK]]],
                    x_case=X[i], x_post=X[i] + dc_ai[0].mean(0), ref_mu=ref.mu, ref_sd=ref.sd,
                    screen_ai=m_ai, screen_random=m_rn)
meta = {'case': CASE, 'theta': theta, 'd0': d0, 'winner': cands[k_best]['name'], 'winner_u': U_ai[k_best].tolist(),
        'confirmed_improvement': float(mc_ai[0]), 'confirmed_ci': [float(np.percentile(bs_ai, 2.5)), float(np.percentile(bs_ai, 97.5))],
        'confirmed_pct_streams_improved': float((pc_ai[0] > 0).mean() * 100),
        'random_winner': '+'.join(CONTROLS[j] for j in np.where(np.abs(U_rn[k_rn]) > 1e-9)[0]),
        'random_confirmed_improvement': float(mc_rn[0]),
        'random_confirmed_ci': [float(np.percentile(bs_rn, 2.5)), float(np.percentile(bs_rn, 97.5))],
        'screen_median_ai': float(np.median(m_ai)), 'screen_median_random': float(np.median(m_rn)),
        'emulator_vs_simulator_r': float(np.corrcoef([c['improvement'] for c in cands], m_ai)[0, 1])}
json.dump(meta, open(f'{R}/single_case_meta.json', 'w'), indent=1)
print(json.dumps(meta, indent=1))
'''

SOURCE_FILES['scripts/fig1_schematic.py'] = r'''"""Figure 1 - model and NeuronAI framework schematic."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
from fig_style import setup, save, panel, C_TXT, C_TXT2, C_SIM, C_TD, C_ASD
from neuronai.model import electrode_positions, EEG_CHANNELS, connectivity_matrices

setup()
C_AMPA, C_N2A, C_N2B = '#1c5cab', '#6250d6', '#9085e9'
C_GA, C_GS = '#e34948', '#eda100'
C_MOD = '#008300'

fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.15, 0.85], hspace=0.12, wspace=0.05)

# ---------------------------------------------------------------- (a) column
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off'); ax.set_aspect('equal')
panel(ax, 'a', x=0.0, y=0.97)
ax.text(0.9, 9.62, 'Extended Wendling column', fontsize=9, fontweight='bold', color=C_TXT)
pops = {'P': (5.0, 5.3, 'Pyramidal\ncells'), 'E': (1.8, 7.6, 'Excitatory\ninterneurons'),
        'S': (1.8, 3.0, 'Slow dendritic\ninterneurons'), 'F': (8.2, 3.0, 'Fast somatic\n(PV-like) int.')}
for k, (x, y, lab) in pops.items():
    col = '#dbe9fa' if k in 'PE' else '#fbe0de'
    ax.add_patch(Circle((x, y), 1.12, fc=col, ec='#52514e', lw=0.8, zorder=3))
    ax.text(x, y, lab, ha='center', va='center', fontsize=5.3, zorder=4, color=C_TXT)


def arrow(p, q, color, lw=1.6, rad=0.0, style='-|>', ls='-'):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                        connectionstyle=f'arc3,rad={rad}', zorder=2, shrinkA=10, shrinkB=10, ls=ls)
    ax.add_patch(a)


P, E, S, F = [pops[k][:2] for k in 'PESF']
arrow(P, E, C_AMPA, rad=0.15)                         # P -> E
arrow(E, P, C_AMPA, rad=0.15)                         # E -> P (AMPA)
arrow((E[0] + 0.3, E[1] - 0.2), (P[0] - 0.2, P[1] + 0.3), C_N2A, rad=-0.05)
arrow((E[0] + 0.5, E[1] - 0.45), (P[0] - 0.45, P[1] + 0.15), C_N2B, rad=-0.15)
arrow(P, S, C_AMPA, rad=-0.15)
arrow(S, P, C_GS, rad=-0.15, style='-[')
arrow(P, F, C_AMPA, rad=0.15)
arrow((P[0] + 0.4, P[1] - 0.6), (F[0] - 0.6, F[1] + 0.6), C_N2A, rad=0.3)
arrow(F, P, C_GA, rad=0.15, style='-[')
arrow(S, F, C_GS, rad=0.0, style='-[')
# drive
ax.annotate('', xy=(P[0] + 0.7, P[1] + 0.75), xytext=(7.4, 7.7),
            arrowprops=dict(arrowstyle='-|>', color=C_AMPA, lw=1.4))
ax.text(6.9, 7.8, 'Drive $p(t)$: tonic $\\bar p$ +\nwhite + slow (OU) noise\n+ long-range input $K$',
        fontsize=5.6, color=C_TXT2, va='bottom')
ax.text(5.0, 0.35, 'EEG source: $v_P = y_1+n_{2A}+n_{2B}-y_2-y_3$', fontsize=5.6, ha='center', color=C_TXT2)
# neuromodulators
mods = [('DA (D1)', 4.0, 9.0, [(P[0] - 0.3, P[1] + 1.0), (F[0] - 0.4, F[1] + 1.0)]),
        ('5-HT$_{2A}$', 5.6, 9.0, [(P[0] + 0.2, P[1] + 1.0)]),
        ('NE', 9.3, 6.0, [(P[0] + 1.0, P[1] + 0.2), (F[0] + 0.2, F[1] + 1.0)])]
for lab, x, y, tg in mods:
    ax.text(x, y, lab, fontsize=6.5, color=C_MOD, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='#e7f5e7', ec=C_MOD, lw=0.6))
    for t in tg:
        ax.annotate('', xy=t, xytext=(x, y - 0.25 if y > 8 else y), arrowprops=dict(arrowstyle='->', color=C_MOD, lw=0.8, ls=(0, (2, 1.5))))
# legend (lines with arrow = excitatory, bar end = inhibitory)
leg = [('AMPA', C_AMPA, 'e'), ('NMDA-GluN2A', C_N2A, 'e'), ('NMDA-GluN2B', C_N2B, 'e'),
       ('GABA$_A$ (fast)', C_GA, 'i'), ('Slow dendritic GABA', C_GS, 'i'), ('Neuromodulation', C_MOD, 'm')]
for i, (lab, c, kind) in enumerate(leg):
    xx = [0.0, 3.1, 7.0][i % 3]; yy = 1.45 - (i // 3) * 0.5
    if kind == 'i':
        ax.plot([xx, xx + 0.7], [yy, yy], color=c, lw=1.6); ax.plot([xx + 0.7] * 2, [yy - 0.15, yy + 0.15], color=c, lw=1.6)
    else:
        ax.annotate('', xy=(xx + 0.75, yy), xytext=(xx, yy), arrowprops=dict(arrowstyle='->' if kind == 'm' else '-|>', color=c, lw=1.4,
                    ls=(0, (2, 1.5)) if kind == 'm' else '-'))
    ax.text(xx + 0.9, yy, lab, fontsize=5.6, va='center', color=C_TXT)
ax.set_ylim(-0.4, 10)
# ---------------------------------------------------------------- (b) network on scalp
ax = fig.add_subplot(gs[0, 1]); ax.set_aspect('equal'); ax.axis('off')
panel(ax, 'b', x=0.0, y=0.97)
ax.text(-1.05, 1.36, '17-column network', fontsize=9, fontweight='bold', color=C_TXT)
pos = electrode_positions()
r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
th = np.arctan2(pos[:, 1], pos[:, 0])
rad = np.arccos(np.clip(pos[:, 2], -1, 1)) / (np.pi / 2)
xy = np.c_[rad * np.cos(th), rad * np.sin(th)]
ax.add_patch(Circle((0, 0), 1.08, fc='#f7f6f3', ec='#9a9994', lw=0.8))
ax.plot([-0.12, 0, 0.12], [1.07, 1.2, 1.07], color='#9a9994', lw=0.8)
W, M = connectivity_matrices(mixing_width=0.5)
for i in range(17):
    for j in range(i + 1, 17):
        w = W[i, j] + W[j, i]
        if w > 0.13:
            ax.plot(*xy[[i, j]].T, color=C_AMPA, lw=2.2 * w, alpha=0.55, zorder=1)
ax.scatter(xy[:, 0], xy[:, 1], s=95, c='#dbe9fa', edgecolors='#1c5cab', linewidths=0.8, zorder=3)
for (x, y), lab in zip(xy, EEG_CHANNELS):
    ax.text(x, y, lab, fontsize=4.6, ha='center', va='center', zorder=4)
ax.text(0, -1.32, 'Coupling $W_{ij}\\propto e^{-d_{ij}/\\ell}$ (long-range, glutamatergic)\n'
        'Sensors $= \\mathbf{M}\\,v_P$ (volume conduction) + average reference',
        fontsize=5.6, ha='center', va='top', color=C_TXT2)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.75, 1.45)

# ---------------------------------------------------------------- (c) pipeline
ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 20); ax.set_ylim(0, 8.2); ax.axis('off')
panel(ax, 'c', x=0.0, y=0.98)
ax.text(0.9, 7.95, 'NeuronAI v2: simulation-based inference and in silico pharmacology', fontsize=9, fontweight='bold')


def box(x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.25', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h - 0.3, title, ha='center', va='top', fontsize=6.0, fontweight='bold', color=C_TXT)
    ax.text(x + w / 2, y + h - 0.9, body, ha='center', va='top', fontsize=4.9, color=C_TXT2, linespacing=1.3)


def arr(p, q):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='-|>', color='#52514e', lw=1.1))


B = '#eef4fc'; Be = '#6da7ec'; G = '#e8f6f0'; Ge = C_SIM; O = '#fdf0ea'; Oe = C_ASD
w, gap, y0, h = 3.6, 0.4, 4.4, 3.0
xs = [0.2 + k * (w + gap) for k in range(5)]
box(xs[0], y0, w, h, '1  Simulation banks', '2 x 8,192 virtual networks\nscrambled Sobol prior\n9 physiological parameters\n(+ neuromodulatory nuisance)', B, Be)
box(xs[1], y0, w, h, '2  EEG descriptors', 'unit-invariant:\nrelative power (\u03b4 \u03b8 \u03b1 \u03b2)\nIAF, 1/f exponent\nPLV\u03b1, wPLI\u03b1, Kuramoto r\u03b1', B, Be)
box(xs[2], y0, w, h, '3  Inverse ensemble', '5 x MLP deep ensemble\nfeatures \u2192 parameters\nsplit-conformal 90 %\nprediction intervals', G, Ge)
box(xs[3], y0, w, h, '4  Digital twin', 'one twin per recording\nparameter estimate + interval\nposterior-predictive and\nout-of-sample checks', G, Ge)
box(xs[4], y0, w, h, '5  Forward emulator', '5 x MLP surrogate\nparameters (+ modulators)\n\u2192 expected descriptors', G, Ge)
for k in range(4):
    arr((xs[k] + w + 0.03, y0 + h / 2), (xs[k + 1] - 0.03, y0 + h / 2))
yb, hb = 0.2, 3.4
box(0.2, yb, 5.4, hb, 'Real EEG (Duville et al.)', 'ASD and TD recordings\n17 shared electrodes, 250 Hz\n1\u201345 Hz, average reference\npre-specified quality control\n20-s windows, split halves', O, Oe)
box(6.0, yb, 5.6, hb, '7  Verification and validation', 'full stochastic simulator, paired seeds\nbootstrap CIs, robustness checks\npharmaco-EEG face validity\nreliability, child identification\nTD leave-one-child-out control', G, Ge)
box(12.0, yb, 7.8, hb, '6  Sparse pharmacological search', 'u \u2208 [\u22121, 1]$^9$: GABA$_A$, slow GABA, AMPA, GluN2A, GluN2B,\ntonic drive, DA, 5-HT, NE; all 129 supports of size \u2264 3\nminimise Mahalanobis distance to the TD reference + \u03bb\u2016u\u2016$^2$\nresidual-preserving prediction of post-intervention EEG\nnull: random u with equal sparsity and norm', G, Ge)
arr((xs[1] + 0.9, yb + hb + 0.05), (xs[1] + 0.9, y0 - 0.05))      # real EEG -> descriptors
arr((xs[3] + w / 2, y0 - 0.05), (xs[3] + w / 2, yb + hb + 0.05))  # twin -> search
arr((xs[4] + w / 2, y0 - 0.05), (xs[4] + w / 2, yb + hb + 0.05))  # emulator -> search
arr((12.0 - 0.03, yb + hb / 2), (11.6 + 0.03, yb + hb / 2))        # search -> verification
save(fig, 'Figure1_framework')
print('ok')
'''

SOURCE_FILES['scripts/09_figures.py'] = r'''"""Step 9 - publication figures (Figures 2-8, Supplementary Figures S1-S3)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fig_style import (setup, save, panel, C_TD, C_ASD, C_SIM, C_NULL, C_TXT, C_TXT2, GRID,
                       DIV_NEG, DIV_MID, DIV_POS, CHILD_ASD, CHILD_TD)
from neuronai.model import FIT_PARAMS, FIT_LABELS, FIT_BOUNDS, CONTROLS, CONTROL_LABELS, params_from_unit
from neuronai.features import FEATURE_NAMES, FEATURE_LABELS
from neuronai.bank import load_bank

setup()
R = 'results'
df_all = pd.read_csv(f'{R}/empirical_features.csv')
df = df_all[~df_all.excluded].reset_index(drop=True)
tw = pd.read_csv(f'{R}/twin_parameters.csv')
emp = np.load(f'{R}/empirical_psd.npz')
ids = list(emp['ids']); fr = emp['freqs']
E = np.stack([emp['logpsd'][ids.index(s)] for s in df.id])
grp = df.group.values
gcol = {'ASD': C_ASD, 'TD': C_TD}
DISP = {'ASD': 'ASD', 'TD': 'TD'}
nice_id = lambda sid: sid.replace('ASD_', 'ASD ').replace('TD_', 'TD ').replace('_restEO', ' rest').replace('_', ' ').replace('EEG', '#')
MANY = df.child.nunique() > 12
children = sorted(df.child.unique())
ccol = {c: (CHILD_ASD if c.startswith('ASD') else CHILD_TD)[k % 4] for k, c in enumerate(children)}
if MANY:
    ccol = {c: gcol[c.split('-')[0]] for c in children}


def child_strip(ax, values, x0, jitter=0.12, s=7):
    rng = np.random.default_rng(0)
    for c in children:
        m = (df.child == c).values
        ax.scatter(x0 + rng.uniform(-jitter, jitter, m.sum()), values[m], s=s, color=ccol[c], alpha=0.75,
                   edgecolors='white', linewidths=0.3, zorder=3)


# =========================================================================== Figure 2
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.42, wspace=0.28)
ax = fig.add_subplot(gs[0, 0]); ax.axis('off'); panel(ax, 'a', x=-0.02, y=1.02)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
n_tot = df_all.groupby('group').size(); n_ok = df.groupby('group').size(); n_ch = df.groupby('group').child.nunique()
ex = df_all[df_all.excluded]
reasons = ex.reason.fillna('').astype(str).str.contains('reference').sum(); badch = ex.reason.fillna('').astype(str).str.contains('bad channels').sum()
def tbox(x, y, w, h, t, fc, ec, fs=6.4):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs)
def why(g):
    e = ex[ex.group == g].reason.fillna('').astype(str)
    return len(e), int(e.str.contains('reference').sum()), int(e.str.contains('bad channels').sum()), int(e.str.contains('clean data').sum())
tbox(0.04, 0.78, 0.92, 0.17, f'{len(df_all):,} EEG recordings · {df_all.child.nunique()} children\n'
     f'ASD: {n_tot["ASD"]} ({df_all[df_all.group == "ASD"].child.nunique()} children) · '
     f'TD: {n_tot["TD"]} ({df_all[df_all.group == "TD"].child.nunique()} children)', '#f7f6f3', '#9a9994', 5.8)
for x0, g, fc, ec in [(0.04, 'ASD', '#fdf0ea', C_ASD), (0.52, 'TD', '#eef4fc', C_TD)]:
    ne, nr, nb, ns = why(g)
    tbox(x0, 0.03, 0.44, 0.26, f'{DISP[g]} analysed\n{n_ok[g]} recordings\n{n_ch[g]} children', fc, ec, 6.0)
    txt = f'excluded: {ne}\nreference failure: {nr}\n> 4 bad channels: {nb}' + (f'\n< 60 s clean: {ns}' if ns else '')
    tbox(x0 + 0.08, 0.36, 0.34, 0.30, txt + '\n(may co-occur)', '#fbe0de', '#e34948', 5.3)
arr = dict(arrowstyle='-|>', color='#52514e', lw=0.9)
for x in (0.08, 0.56):
    ax.annotate('', xy=(x, 0.29), xytext=(x, 0.78), arrowprops=arr)
ax.set_title('Recordings and quality control', loc='left', pad=6, x=0.06)
ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b')
for g in ['TD', 'ASD']:
    m = grp == g
    mu, sd = E[m].mean(0), E[m].std(0)
    ax.fill_between(fr, mu - sd, mu + sd, color=gcol[g], alpha=0.18, lw=0)
    ax.plot(fr, mu, color=gcol[g], lw=1.6, label=f'{DISP[g]} (n = {m.sum()} recordings)')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('log$_{10}$ relative power')
ax.set_title('Empirical spectra (mean ± SD)'); ax.legend(loc='upper right')
ax.set_xlim(1, 30)

ax = fig.add_subplot(gs[1, :]); panel(ax, 'c', x=-0.07)
bank = load_bank(f'{R}/bank_I.npz'); BX = bank['X'][bank['ok']]
mu, sd = BX.mean(0), BX.std(0)
Z = (BX - mu) / sd
parts = ax.violinplot([Z[:, j] for j in range(len(FEATURE_NAMES))], positions=np.arange(len(FEATURE_NAMES)),
                      widths=0.8, showextrema=False)
for b in parts['bodies']:
    b.set_facecolor('#d9d8d3'); b.set_edgecolor('none'); b.set_alpha(1)
rng = np.random.default_rng(1)
for j, f in enumerate(FEATURE_NAMES):
    for g, off in [('ASD', -0.13), ('TD', 0.13)]:
        v = (df[f].values[grp == g] - mu[j]) / sd[j]
        ax.scatter(j + off + rng.uniform(-0.05, 0.05, len(v)), v, s=3 if MANY else 5, color=gcol[g], alpha=0.5 if MANY else 0.7, lw=0, zorder=3,
                   label=DISP[g] if j == 0 else None)
    q = [(BX[:, j] < v).mean() for v in df[f].values]
    ax.text(j, 4.2, f'{100*np.mean([(0.025 < qq < 0.975) for qq in q]):.0f}%', ha='center', fontsize=6.3, color=C_TXT2)
ax.text(-0.6, 4.75, 'Recordings inside the central 95 % of the prior predictive:', fontsize=6.3, color=C_TXT2)
ax.set_xticks(range(len(FEATURE_NAMES))); ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], rotation=0, fontsize=6.4)
ax.set_ylabel('z-score relative to simulated bank'); ax.set_ylim(-4.5, 5.2)
ax.set_title('Prior predictive check: 8,192 simulations (gray) vs. real recordings', pad=16)
ax.legend(loc='lower right', ncol=2, markerscale=2)
save(fig, 'Figure2_data_priorpredictive')

# =========================================================================== Figure 3
rec = pd.read_csv(f'{R}/recovery_inverse.csv').set_index('parameter')
z = np.load(f'{R}/recovery_test.npz')
fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6))
cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#104281'])
for k, (ax, n) in enumerate(zip(axes.ravel(), FIT_PARAMS)):
    lo, hi = FIT_BOUNDS[n]
    t = lo + z['true'][:, k] * (hi - lo); p = lo + z['pred'][:, k] * (hi - lo)
    ax.hexbin(t, p, gridsize=28, cmap=cmap, mincnt=1, linewidths=0.1, extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], color=C_TXT2, lw=0.8, ls='--')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    r = rec.loc[n]
    ax.text(0.04, 0.96, f"$R^2$ = {r.R2:.2f}\ncoverage = {100*r.conformal_coverage_90:.0f}%",
            transform=ax.transAxes, va='top', fontsize=6.5, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
    ax.set_title(FIT_LABELS[n], fontsize=8)
    if k % 3 == 0: ax.set_ylabel('Estimated')
    if k >= 6: ax.set_xlabel('Ground truth')
n_test = json.load(open(f'{R}/training_summary.json'))['n_test_inv']
fig.suptitle(f'Parameter recovery on {n_test:,} held-out simulations (inverse deep ensemble, 90 % conformal intervals)',
             fontsize=8.5, fontweight='bold', x=0.02, ha='left', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, 'Figure3_identifiability')

# =========================================================================== Figure 4
ppc = np.load(f'{R}/ppc.npz', allow_pickle=True)
S = ppc['LP_full']
ppcs = pd.read_csv(f'{R}/ppc_summary.csv').set_index('feature')
r2t = pd.read_csv(f'{R}/ppc_spectral_r2.csv')
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.05, 1.05], hspace=0.7, wspace=0.55)
med = r2t.groupby('group').R2_twin.median()
examples = []
for g in ['ASD', 'TD']:
    sub = r2t[r2t.group == g].sort_values('R2_twin')
    examples += [sub.iloc[len(sub) // 2].id, sub.iloc[int(len(sub) * 0.15)].id]
for k, sid in enumerate(examples):
    ax = fig.add_subplot(gs[0, k]); i = int(np.where(df.id == sid)[0][0])
    if k == 0: panel(ax, 'a', x=-0.35)
    g = df.group[i]
    lo_, hi_ = np.percentile(S[i], [2.5, 97.5], axis=0)
    ax.fill_between(fr, lo_, hi_, color=C_SIM, alpha=0.25, lw=0)
    ax.plot(fr, S[i].mean(0), color=C_SIM, lw=1.3, label='Twin (simulated)')
    ax.plot(fr, E[i], color=gcol[g], lw=1.3, label='Recorded')
    rr = r2t.set_index('id').loc[sid, 'R2_twin']
    ax.set_title(nice_id(sid), fontsize=7)
    ax.text(0.97, 0.95, f'$R^2$={rr:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
    ax.set_xlabel('Hz'); ax.set_xlim(1, 30)
    if k == 0:
        ax.set_ylabel('log$_{10}$ rel. power')
        ax.legend(loc='lower left', fontsize=5.8, handlelength=1.2)
ax = fig.add_subplot(gs[1, 0:2]); panel(ax, 'b', x=-0.2, y=1.12)
r2a = pd.read_csv(f'{R}/ppc_spectral_r2_amortised.csv').set_index('id')
for j, g in enumerate(['ASD', 'TD']):
    sub = r2t[r2t.group == g]
    x0 = j * 3.6
    vals = [sub.R2_group_mean.values, r2a.loc[sub.id, 'R2_amortised'].values, sub.R2_twin.values]
    cols = [C_NULL, '#9ec5f4', gcol[g]]
    for k2, (v, c) in enumerate(zip(vals, cols)):
        rng = np.random.default_rng(k2)
        ax.scatter(x0 + k2 + rng.uniform(-0.18, 0.18, len(v)), np.clip(v, -1, 1), s=5, color=c, alpha=0.8, lw=0, zorder=3)
        ax.plot([x0 + k2 - 0.3, x0 + k2 + 0.3], [np.median(v)] * 2, color=C_TXT, lw=1.4, zorder=4)
ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6])
ax.set_xticklabels(['group\nmean', 'amor-\ntised', 'refined\ntwin'] * 2, fontsize=5.8)
ax.text(1, 1.06, 'ASD', transform=ax.get_xaxis_transform(), ha='center', color=C_ASD, fontweight='bold')
ax.text(4.6, 1.06, 'TD', transform=ax.get_xaxis_transform(), ha='center', color=C_TD, fontweight='bold')
ax.set_ylabel('Spectral $R^2$ (clipped at −1)'); ax.set_ylim(-1.05, 1.0)
ax.set_title('Spectral fit, 1–30 Hz (bars: medians)', pad=14)
ax = fig.add_subplot(gs[1, 2:]); panel(ax, 'c', x=-0.16)
xx = np.arange(len(FEATURE_NAMES))
ax.bar(xx - 0.27, ppcs.r_in_sample, 0.26, color='#9ec5f4', label='in-sample')
ax.bar(xx, ppcs.r_out_of_sample, 0.26, color=C_SIM, label='out-of-sample (half 1 → half 2)')
ax.bar(xx + 0.27, ppcs.r_test_retest, 0.26, color='#d9d8d3', label='test–retest ceiling')
ax.set_xticks(xx); ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], rotation=45, ha='right', fontsize=6)
ax.set_ylabel('Pearson r'); ax.axhline(0, color='#9a9994', lw=0.6)
ax.legend(fontsize=5.6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0, columnspacing=0.8); ax.set_ylim(min(-0.2, ppcs[['r_in_sample', 'r_out_of_sample']].min().min() - 0.05), 1.05)
ax.set_title('Recorded vs. predicted descriptors', pad=16)
F_oos = ppc['F_h1'].mean(1)
X2 = df[[f + '_h2' for f in FEATURE_NAMES]].values
for k, f in enumerate(['rel_alpha', 'iaf', 'exponent', 'kuramoto_alpha']):
    ax = fig.add_subplot(gs[2, k]); j = FEATURE_NAMES.index(f)
    if k == 0: panel(ax, 'd', x=-0.35)
    for g in ['ASD', 'TD']:
        m = grp == g
        ax.scatter(F_oos[m, j], X2[m, j], s=3 if MANY else 6, color=gcol[g], alpha=0.5 if MANY else 0.75, lw=0, label=DISP[g])
    lim = [min(F_oos[:, j].min(), X2[:, j].min()), max(F_oos[:, j].max(), X2[:, j].max())]
    pad = 0.05 * (lim[1] - lim[0]); lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, color=C_TXT2, ls='--', lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(FEATURE_LABELS[f], fontsize=7.5)
    ax.text(0.04, 0.96, f"r = {ppcs.loc[f, 'r_out_of_sample']:.2f}", transform=ax.transAxes, va='top', fontsize=6.5)
    ax.set_xlabel('Twin prediction (half 1)', fontsize=6.5)
    if k == 0:
        ax.set_ylabel('Recorded (half 2)', fontsize=6.5); ax.legend(fontsize=5.8, loc='lower right', handletextpad=0.1)
save(fig, 'Figure4_validation_real_eeg')

# =========================================================================== Figure 5
rel = pd.read_csv(f'{R}/reliability.csv')
vs = json.load(open(f'{R}/validation_summary.json'))
fig = plt.figure(figsize=(7.2, 5.8))
gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.75)
ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a', x=-0.75)
rp = rel[~rel.parameter.str.startswith('feature:')].reset_index(drop=True)
yy = np.arange(len(rp))[::-1]
ax.axvspan(0.75, 1.0, color='#e8f6f0', lw=0); ax.axvspan(0.5, 0.75, color='#f4faf7', lw=0)
ax.errorbar(rp.ICC_split_half, yy, xerr=[rp.ICC_split_half - rp.ci_lo, rp.ci_hi - rp.ICC_split_half], fmt='o',
            color=C_SIM, ms=4, elinewidth=1, capsize=0)
ax.set_yticks(yy); ax.set_yticklabels([FIT_LABELS[n] for n in rp.parameter], fontsize=6.3)
ax.set_xlabel('ICC(2,1), split halves'); ax.set_xlim(-0.2, 1.0)
ax.set_title('Split-half reliability')
ax = fig.add_subplot(gs[0, 1:]); panel(ax, 'b', x=-0.1)
show = ['p', 'K', 'rho', 'A', 'B']
SHORT = {'p': 'Drive $\\bar p$', 'K': 'Coupling $K$', 'B': 'Slow inh. $B$', 'rho': 'Slow input $\\rho$', 'A': 'AMPA $A$'}
if MANY:
    # child medians (dots) and within-child interquartile ranges (bars), children sorted within each parameter
    for k, n in enumerate(show):
        lo, hi = FIT_BOUNDS[n]
        vv = (tw[f'{n}_full'].values - lo) / (hi - lo)
        st = pd.DataFrame({'v': vv, 'child': df.child.values}).groupby('child').v.describe()
        st = st.sort_values('50%')
        xs = k + np.linspace(-0.4, 0.4, len(st))
        for x_, (c, r_) in zip(xs, st.iterrows()):
            col = gcol[c.split('-')[0]]
            ax.plot([x_, x_], [r_['25%'], r_['75%']], color=col, lw=0.6, alpha=0.55)
            ax.scatter([x_], [r_['50%']], s=4, color=col, lw=0, zorder=3)
        icc1 = rel.set_index('parameter').loc[n, 'ICC1_between_child']
        ax.text(k, 1.06, f'ICC(1) = {icc1:.2f}', ha='center', fontsize=6.0, color=C_TXT2)
        if k: ax.axvline(k - 0.5, color=GRID, lw=0.8)
    for g in ['ASD', 'TD']:
        ax.scatter([], [], color=gcol[g], s=12, label=f'{DISP[g]} child (median, IQR)')
    ax.legend(ncol=2, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, -0.3))
    ax.set_title(f'Twin parameters of {len(children)} children (sorted within each parameter)', pad=14)
else:
    offs = np.linspace(-0.36, 0.36, len(children))
    rng = np.random.default_rng(0)
    for k, n in enumerate(show):
        v = tw[f'{n}_full'].values
        lo, hi = FIT_BOUNDS[n]
        vv = (v - lo) / (hi - lo)
        for o, c in zip(offs, children):
            m = (df.child == c).values
            ax.scatter(k + o + rng.uniform(-0.025, 0.025, m.sum()), vv[m], s=5, color=ccol[c], alpha=0.8, lw=0, zorder=3)
            ax.plot([k + o - 0.04, k + o + 0.04], [np.median(vv[m])] * 2, color=C_TXT, lw=1.0, zorder=4)
        icc1 = rel.set_index('parameter').loc[n, 'ICC1_between_child']
        ax.text(k, 1.06, f'ICC(1) = {icc1:.2f}', ha='center', fontsize=6.0, color=C_TXT2)
        if k: ax.axvline(k - 0.5, color=GRID, lw=0.8)
    for c in children:
        ax.scatter([], [], color=ccol[c], s=12, label=c)
    ax.legend(ncol=8, fontsize=5.8, loc='lower center', bbox_to_anchor=(0.5, -0.3), handletextpad=0.05, columnspacing=0.7)
    ax.set_title('Twin parameters by child (each column = one child; ticks = medians)', pad=14)
ax.set_xticks(range(len(show))); ax.set_xticklabels([SHORT[n] for n in show], fontsize=6.3)
ax.set_ylabel('Twin estimate (prior range = 0–1)'); ax.set_ylim(-0.05, 1.13); ax.grid(axis='x', visible=False)
ax = fig.add_subplot(gs[1, 0]); panel(ax, 'c', x=-0.45)
ci = vs['child_identification']; cd = vs.get('child_identification_discriminant')
vals = [ci['perm_mean'], ci['accuracy']]
labs = ['permuted labels', 'twin, centroid']
cols = ['#d9d8d3', C_SIM]
if cd is not None:
    vals += [cd['accuracy'], cd['reference_raw_descriptors_discriminant']]
    labs += ['twin, discriminant', 'raw EEG (reference)']
    cols += [C_SIM, '#9a9994']
    for g in ['ASD', 'TD']:
        w = cd.get(f'within_{g}')
        if w:
            vals.append(w['accuracy_discriminant']); labs.append(f'twin, within {DISP[g]}'); cols.append(gcol[g])
xs = np.arange(len(vals))
ax.bar(xs, vals, color=cols, width=0.62)
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=5.4, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel('Child identification accuracy'); ax.set_ylim(0, 1.08)
for x_, v_ in zip(xs[1:], vals[1:]):
    ax.text(x_, v_ + 0.03, f"{100*v_:.0f}%", ha='center', fontsize=5.8)
ax.text(0.98, 0.97, f"{ci['n_children']} children\nchance {100/ci['n_children']:.1f}%", transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color=C_TXT2)
ax.set_title('Which child?')
gp = pd.read_csv(f'{R}/group_params.csv').set_index('variable')
ax = fig.add_subplot(gs[1, 1:]); panel(ax, 'd', x=-0.3)
order = FIT_PARAMS
yy = np.arange(len(order))[::-1]
g_ = gp.loc[order]
lim_ = max(2.0, np.nanmax(np.abs(g_[['g_ci_lo', 'g_ci_hi']].values)) + 0.3)
ax.axvline(0, color='#9a9994', lw=0.7)
ax.errorbar(g_.hedges_g, yy, xerr=[g_.hedges_g - g_.g_ci_lo, g_.g_ci_hi - g_.hedges_g], fmt='o', color=C_ASD, ms=4, elinewidth=1)
for y_, (_, rw) in zip(yy, g_.iterrows()):
    ax.text(lim_ * 1.02, y_, f"q = {rw.q_fdr:.3f}", va='center', fontsize=6, color=C_TXT2)
ax.set_yticks(yy); ax.set_yticklabels([FIT_LABELS[n] for n in order], fontsize=6.3)
ax.set_xlabel("Hedges' g, ASD – TD (cluster-bootstrap 95 % CI)"); ax.set_xlim(-lim_, lim_ * 1.35)
ax.set_title('ASD vs. TD (confounded with device and sex; child-level p)')
save(fig, 'Figure5_reliability_personalisation')

# =========================================================================== Figure 6
pv = pd.read_csv(f'{R}/pharmaco_validity.csv')
eff = pd.read_csv(f'{R}/pharmaco_effects_all.csv')
fp = pd.read_csv(f'{R}/control_fingerprints.csv')
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, len(pv), height_ratios=[1, 1.05], hspace=0.62, wspace=0.75)
for k, (_, rw) in enumerate(pv.iterrows()):
    ax = fig.add_subplot(gs[0, k])
    v = eff[eff.drug == rw.drug][rw.feature].values
    gg = eff[eff.drug == rw.drug].group.values
    rng = np.random.default_rng(k)
    for g, off in [('ASD', -0.15), ('TD', 0.15)]:
        ax.scatter(off + rng.uniform(-0.08, 0.08, (gg == g).sum()), v[gg == g], s=3 if MANY else 5, color=gcol[g], alpha=0.4 if MANY else 0.6, lw=0, label=DISP[g])
    ax.errorbar([0.5], [rw.mean_change], yerr=[[rw.mean_change - rw.ci_lo], [rw.ci_hi - rw.mean_change]], fmt='D',
                color=C_TXT, ms=4, capsize=2, lw=1)
    ax.axhline(0, color='#9a9994', lw=0.7)
    exp_up = rw.expected_sign == '+'
    ax.annotate('', xy=(0.86, 0.93 if exp_up else 0.66), xytext=(0.86, 0.66 if exp_up else 0.93), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=C_SIM if rw.concordant else '#e34948', lw=1.6))
    ax.text(0.86, 0.98, 'expected', transform=ax.transAxes, ha='center', fontsize=5.3, color=C_TXT2)
    name = rw.drug.split(' (')[0].replace('GABA_A', 'GABA$_A$').replace('5-HT2A', '5-HT$_{2A}$')
    mech = '(' + rw.drug.split(' (')[1]
    import textwrap
    ax.set_title('\n'.join(textwrap.wrap(name, 18)) + f'\n{mech}', fontsize=5.7)
    lab = {'rel_beta': 'Δ log β rel. power', 'rel_alpha': 'Δ log α rel. power', 'theta_beta': 'Δ log θ/β ratio',
           'rel_delta': 'Δ log δ rel. power'}[rw.feature]
    ax.set_ylabel(lab, fontsize=6.3); ax.set_xticks([]); ax.set_xlim(-0.45, 1.05)
    verdict = 'reproduced' if rw.concordant else ('opposite' if (rw.ci_lo > 0 or rw.ci_hi < 0) else 'not significant')
    ax.text(0.5, -0.09, f"{rw.pct_twins_expected_direction:.0f}% of twins in\nexpected direction:\n{verdict}", transform=ax.transAxes, ha='center',
            va='top', fontsize=5.8, color=C_SIM if rw.concordant else '#e34948')
    if k == 0: ax.legend(fontsize=5.5, loc='lower left', handletextpad=0.1)
    panel(ax, 'abcde'[k], x=-0.55, y=1.16)
ax = fig.add_subplot(gs[1, :]); panel(ax, 'f', x=-0.08)
feats = FEATURE_NAMES
sdx = df[(df.group == 'TD')][feats].std().values
M = np.zeros((len(CONTROLS), len(feats))); sig = np.zeros_like(M, dtype=bool)
for i, c in enumerate(CONTROLS):
    for j, f in enumerate(feats):
        rw = fp[(fp.control == c) & (fp.feature == f)].iloc[0]
        M[i, j] = rw['mean'] / sdx[j]; sig[i, j] = (rw.ci_lo > 0) or (rw.ci_hi < 0)
cm = LinearSegmentedColormap.from_list('div', [DIV_NEG, DIV_MID, DIV_POS])
vmax = np.nanpercentile(np.abs(M), 95)
im = ax.imshow(M, cmap=cm, vmin=-vmax, vmax=vmax, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:+.1f}' + ('' if sig[i, j] else '°'), ha='center', va='center', fontsize=5.6,
                color='white' if abs(M[i, j]) > 0.6 * vmax else C_TXT)
ax.set_xticks(range(len(feats))); ax.set_xticklabels([FEATURE_LABELS[f] for f in feats], fontsize=6.2)
ax.set_yticks(range(len(CONTROLS))); ax.set_yticklabels([CONTROL_LABELS[c] + ' (+1)' for c in CONTROLS], fontsize=6.2)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label('Δ descriptor (TD SD units)', fontsize=6.3); cb.ax.tick_params(labelsize=6)
ax.set_title('Mechanistic fingerprints of each control across all twins (° = cluster-bootstrap CI includes 0)')
save(fig, 'Figure7_pharmacological_face_validity')

# =========================================================================== Figure 7 single case
sc = np.load(f'{R}/single_case.npz')
top = pd.read_csv(f'{R}/single_case_top10.csv')
meta = json.load(open(f'{R}/single_case_meta.json'))
fig = plt.figure(figsize=(7.2, 5.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.45, 1], hspace=0.5, wspace=0.55)
ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a')
td = sc['emp_td']; mu, sd = td.mean(0), td.std(0)
ax.fill_between(sc['freqs'], mu - sd, mu + sd, color=C_TD, alpha=0.18, lw=0, label='TD reference (mean ± SD)')
ax.plot(sc['freqs'], mu, color=C_TD, lw=1.2)
ax.plot(sc['freqs'], sc['emp_case'], color=C_ASD, lw=1.4, label=f"{nice_id(meta['case'])} (recorded)")
post = sc['emp_case'] + sc['lp_best'].mean(0) - sc['lp_base'].mean(0)
ax.plot(sc['freqs'], post, color=C_SIM, lw=1.4, ls='--', label='predicted after the confirmed top intervention')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('log$_{10}$ relative power'); ax.set_xlim(1, 30)
ax.legend(fontsize=5.6, loc='upper right', framealpha=0.9); ax.set_title('Single case vs. TD reference')
ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b')
zc = (sc['x_case'] - sc['ref_mu']) / sc['ref_sd']; zp = (sc['x_post'] - sc['ref_mu']) / sc['ref_sd']
yy = np.arange(len(FEATURE_NAMES))[::-1]
ax.axvspan(-1, 1, color='#eef4fc', lw=0); ax.axvline(0, color='#9a9994', lw=0.7)
for y_, a, b in zip(yy, zc, zp):
    ax.annotate('', xy=(b, y_), xytext=(a, y_), arrowprops=dict(arrowstyle='-|>', color=C_SIM, lw=1.0))
ax.scatter(zc, yy, color=C_ASD, s=18, zorder=3, label='before'); ax.scatter(zp, yy, color=C_SIM, s=18, zorder=3, label='after (predicted)')
ax.set_yticks(yy); ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], fontsize=6.3)
ax.set_xlabel('z-score relative to TD reference'); ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
ax.set_title('Descriptor profile (top-1 intervention)')
ax = fig.add_subplot(gs[1, 0]); panel(ax, 'c', x=-0.55)
yy = np.arange(len(top))[::-1]
ax.errorbar(top.improvement_simulated, yy, xerr=[top.improvement_simulated - top.ci_lo, top.ci_hi - top.improvement_simulated],
            fmt='o', color=C_SIM, ms=4, elinewidth=1.2, capsize=2, label='full simulator (20 paired streams)')
ax.scatter(top.improvement_emulated, yy, marker='x', color=C_TXT2, s=16, label='emulator estimate', zorder=3)
ax.axvline(0, color='#9a9994', lw=0.7)
PRETTY = lambda t: t.replace('GABA_slow', 'slow GABA').replace('GABA_A', 'GABA$_A$').replace('NMDA_2A', 'GluN2A').replace('NMDA_2B', 'GluN2B').replace('5HT', '5-HT$_{2A}$').replace(' / ', ', ')
ax.set_yticks(yy); ax.set_yticklabels([f'{r}. {PRETTY(u)}' for r, u in zip(top['rank'], top['u'])], fontsize=5.6)
ax.set_xlabel('Distance reduction to TD (%)')
ax.legend(fontsize=5.6, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); ax.set_title('Ten best simulator-verified interventions')
ax = fig.add_subplot(gs[1, 1]); panel(ax, 'd', x=-0.2)
rng = np.random.default_rng(3)
for k2, (v, c, lab) in enumerate([(sc['screen_random'], C_NULL, 'random\n(matched)'), (sc['screen_ai'], C_SIM, 'NeuronAI\ncandidates')]):
    ax.scatter(k2 + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=c, alpha=0.85, lw=0)
    ax.plot([k2 - 0.25, k2 + 0.25], [np.median(v)] * 2, color=C_TXT, lw=1.4)
ax.errorbar([2.1], [meta['confirmed_improvement']], yerr=[[meta['confirmed_improvement'] - meta['confirmed_ci'][0]],
            [meta['confirmed_ci'][1] - meta['confirmed_improvement']]], fmt='D', color=C_SIM, ms=5, capsize=2)
ax.errorbar([2.5], [meta['random_confirmed_improvement']], yerr=[[meta['random_confirmed_improvement'] - meta['random_confirmed_ci'][0]],
            [meta['random_confirmed_ci'][1] - meta['random_confirmed_improvement']]], fmt='D', color=C_NULL, ms=5, capsize=2)
ax.set_xticks([0, 1, 2.3]); ax.set_xticklabels(['random\n(matched)', 'NeuronAI\ncandidates', 'winners on\nfresh streams'], fontsize=6)
ax.axhline(0, color='#9a9994', lw=0.7); ax.set_ylabel('Distance reduction (%)')
ax.set_title('Guided vs. random search (same budget)')
save(fig, 'Figure9_single_case')

# =========================================================================== Figure 8 intervention across recordings
iv = pd.read_csv(f'{R}/intervention_results.csv')
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1.25], hspace=0.95, wspace=0.85)
ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a', x=-0.35)
for j, g in enumerate(['ASD', 'TD']):
    sub = iv[iv.group == g]
    if MANY:   # one point per child: median over that child's recordings
        sub = sub.groupby('child')[['random_improvement_confirmed', 'improvement_confirmed']].median().reset_index()
    x0 = j * 2.4
    for _, rw in sub.iterrows():
        ax.plot([x0, x0 + 1], [rw.random_improvement_confirmed, rw.improvement_confirmed], color=gcol[g], alpha=0.18, lw=0.6)
    ax.scatter(np.full(len(sub), x0), sub.random_improvement_confirmed, s=6, color=C_NULL, zorder=3, lw=0)
    ax.scatter(np.full(len(sub), x0 + 1), sub.improvement_confirmed, s=6, color=gcol[g], zorder=3, lw=0)
    for xx_, v in [(x0, sub.random_improvement_confirmed), (x0 + 1, sub.improvement_confirmed)]:
        ax.plot([xx_ - 0.25, xx_ + 0.25], [np.median(v)] * 2, color=C_TXT, lw=1.5, zorder=4)
ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(['random', 'Neuron-\nAI', 'random', 'Neuron-\nAI'], fontsize=5.8)
ax.text(0.5, 1.03, 'ASD', transform=ax.get_xaxis_transform(), ha='center', color=C_ASD, fontweight='bold')
ax.text(2.9, 1.03, 'TD (LOCO)', transform=ax.get_xaxis_transform(), ha='center', color=C_TD, fontweight='bold')
lo_ = np.percentile(np.r_[iv.random_improvement_confirmed, iv.improvement_confirmed], 1)
ax.set_ylim(max(lo_ - 5, -100), None)
ax.axhline(0, color='#9a9994', lw=0.6); ax.set_ylabel(('Child median of confirmed\nreduction (%)') if MANY else 'Confirmed distance reduction (%)')
ax.set_title('Guided vs. random', pad=14)
ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b', x=-0.5)
ax.scatter(iv.improvement_emulated, iv.improvement_confirmed, c=[gcol[g] for g in iv.group], s=7, alpha=0.75, lw=0)
lim = [min(iv.improvement_emulated.min(), iv.improvement_confirmed.min()) - 3, max(iv.improvement_emulated.max(), iv.improvement_confirmed.max()) + 3]
lim[0] = max(lim[0], -60)
ax.plot(lim, lim, color=C_TXT2, ls='--', lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Emulator estimate (%)'); ax.set_ylabel('Simulator, confirmed (%)')
ax.text(0.04, 0.96, f"r = {vs['intervention']['emulated_vs_confirmed_r']:.2f}", transform=ax.transAxes, va='top', fontsize=6.5)
ax.set_title('Emulator vs. simulator', fontsize=8)
a = iv[iv.group == 'ASD']
ax = fig.add_subplot(gs[0, 2]); panel(ax, 'c', x=-0.42)
sel = a[[f'u_{c}' for c in CONTROLS]]
up = (sel > 1e-6).mean(); dn = (sel < -1e-6).mean()
xx = np.arange(len(CONTROLS))
ax.bar(xx, up * 100, color=DIV_POS, width=0.65, label='increase (agonism)')
ax.bar(xx, -dn * 100, color=DIV_NEG, width=0.65, label='decrease (antagonism)')
ax.axhline(0, color='#52514e', lw=0.6)
ax.set_xticks(xx); ax.set_xticklabels([CONTROL_LABELS[c] for c in CONTROLS], rotation=60, ha='right', fontsize=6)
ax.set_ylabel('% of ASD recordings'); ax.legend(fontsize=5.4, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, columnspacing=0.6)
ax.set_title('Selected targets', pad=18)
ax = fig.add_subplot(gs[1, :]); panel(ax, 'd', x=-0.06)
a_sorted = a.sort_values(['child', 'condition'])
M = a_sorted[[f'u_{c}' for c in CONTROLS]].values.T
cm = LinearSegmentedColormap.from_list('div', [DIV_NEG, DIV_MID, DIV_POS])
im = ax.imshow(M, aspect='auto', cmap=cm, vmin=-1, vmax=1, interpolation='nearest')
ax.set_yticks(range(len(CONTROLS))); ax.set_yticklabels([CONTROL_LABELS[c] for c in CONTROLS], fontsize=6.3)
ax.grid(False)
bounds = np.cumsum(a_sorted.groupby('child').size().values)
for b in bounds[:-1]:
    ax.axvline(b - 0.5, color='#0b0b0b', lw=0.4 if MANY else 0.8)
starts = np.r_[0, bounds[:-1]]
if not MANY:
    for c, s0, s1 in zip(a_sorted.child.unique(), starts, bounds):
        ax.text((s0 + s1 - 1) / 2, -0.9, c, ha='center', fontsize=6.8, color=C_ASD, fontweight='bold')
ax.set_xticks([]); ax.set_xlabel(f'ASD recordings, grouped by child ({a_sorted.child.nunique()} children; vertical lines separate children)')
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label('control u', fontsize=6.5); cb.ax.tick_params(labelsize=6)
ax.set_title(f"Selected intervention vectors (within-child cosine {vs['intervention']['u_cosine_within_child']:.2f} vs. "
             f"between-child {vs['intervention']['u_cosine_between_child']:.2f}; permutation p {'< 0.001' if vs['intervention']['u_cosine_perm_p'] < 0.001 else '= %.3f' % vs['intervention']['u_cosine_perm_p']})", pad=14)
save(fig, 'Figure8_intervention_across_recordings')

# =========================================================================== Supplementary
nc = pd.read_csv(f'{R}/noise_ceiling.csv').set_index('feature')
ea = pd.read_csv(f'{R}/emulator_accuracy.csv').set_index('feature')
fig, ax = plt.subplots(figsize=(4.6, 2.6))
xx = np.arange(len(FEATURE_NAMES))
ax.bar(xx - 0.2, nc.loc[FEATURE_NAMES, 'noise_ceiling_R2'], 0.38, color='#d9d8d3', label='noise ceiling (4 seeds × 128 parameter sets)')
ax.bar(xx + 0.2, ea.loc[FEATURE_NAMES, 'R2_heldout'], 0.38, color=C_SIM, label='forward emulator (held-out simulations)')
ax.set_xticks(xx); ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], rotation=45, ha='right', fontsize=6.3)
ax.set_ylabel('$R^2$'); ax.set_ylim(0, 1.08); ax.legend(fontsize=6, loc='lower left')
save(fig, 'FigureS1_emulator_noise_ceiling')

fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.6))
for ax, f in zip(axes.ravel(), FEATURE_NAMES + ['_']):
    if f == '_':
        ax.axis('off'); continue
    for k, c in enumerate(children):
        v = df[df.child == c][f].values
        ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=2 if MANY else 5, color=ccol[c], lw=0)
    if MANY:
        ax.set_xticks([]); ax.set_xlabel('children (ASD | TD)', fontsize=6)
    else:
        ax.set_xticks(range(len(children))); ax.set_xticklabels([c.replace('-', '\n') for c in children], fontsize=5)
    ax.set_title(FEATURE_LABELS[f], fontsize=7)
fig.tight_layout()
save(fig, 'FigureS2_features_by_child')
print('figures done')
'''

SOURCE_FILES['scripts/fig_graphical_abstract.py'] = r'''"""Graphical abstract (1328 x 531 px minimum; produced at 600 dpi)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from fig_style import setup, save, C_TD, C_ASD, C_SIM, C_TXT, C_TXT2

setup()
R = 'results'
df = pd.read_csv(f'{R}/empirical_features.csv'); df = df[~df.excluded].reset_index(drop=True)
emp = np.load(f'{R}/empirical_psd.npz'); ids = list(emp['ids']); fr = emp['freqs']
ppc = np.load(f'{R}/ppc.npz', allow_pickle=True)
vs = json.load(open(f'{R}/validation_summary.json'))
rec = pd.read_csv(f'{R}/recovery_inverse.csv')

fig = plt.figure(figsize=(13.28 / 2.0, 5.31 / 2.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis('off')
def card(x, w, title, col):
    ax.add_patch(FancyBboxPatch((x, 7.5), w, 29, boxstyle='round,pad=0.3,rounding_size=1.2', fc='white', ec=col, lw=1.2))
    ax.text(x + w / 2, 34.6, title, ha='center', va='center', fontsize=7.2, fontweight='bold', color=C_TXT)
card(1.5, 27, 'Scalp EEG of children', C_ASD)
card(36, 28, 'Biophysical digital twin', C_SIM)
card(71.5, 27, 'In silico pharmacology', '#6250d6')
for x0, x1 in [(29.2, 35.4), (64.7, 70.9)]:
    ax.annotate('', xy=(x1, 22), xytext=(x0, 22), arrowprops=dict(arrowstyle='-|>', color='#52514e', lw=1.4))
# panel 1: real spectra
a1 = fig.add_axes([0.075, 0.37, 0.195, 0.42])
E = emp['logpsd']; grp = np.array(['ASD' if s.startswith('ASD') else 'TD' for s in ids])
keep = np.isin(ids, df.id.values)
for g, c in [('TD', C_TD), ('ASD', C_ASD)]:
    m = (grp == g) & keep
    a1.plot(fr, E[m].mean(0), color=c, lw=1.4, label=g)
    a1.fill_between(fr, E[m].mean(0) - E[m].std(0), E[m].mean(0) + E[m].std(0), color=c, alpha=0.15, lw=0)
a1.set_ylabel('log rel. power', fontsize=5.5); a1.tick_params(labelsize=5)
a1.legend(fontsize=5.5, loc='upper right'); a1.set_xlim(1, 30)
ax.text(15, 9.0, f'{len(df):,} recordings · {df.child.nunique()} children · 1–30 Hz', ha='center', fontsize=5.6, color=C_TXT2)
# panel 2: twin fit example + text
a2 = fig.add_axes([0.40, 0.42, 0.20, 0.36])
sid = df.id[0]; i = 0
S = ppc['LP_full'][i]
a2.fill_between(fr, np.percentile(S, 2.5, 0), np.percentile(S, 97.5, 0), color=C_SIM, alpha=0.25, lw=0)
a2.plot(fr, S.mean(0), color=C_SIM, lw=1.3, label='twin')
a2.plot(fr, E[ids.index(sid)], color=C_ASD, lw=1.2, label='recorded')
a2.legend(fontsize=5.3, loc='lower left'); a2.set_xticks([]); a2.set_yticks([]); a2.set_xlim(1, 30)
ax.text(50, 14.2, 'AMPA · NMDA-GluN2A/2B · GABA$_A$ · slow GABA\nDA · 5-HT · NE gain control · 17 coupled columns',
        ha='center', va='center', fontsize=5.4, color=C_TXT2)
ax.text(50, 9.6, 'simulation-based inference:\ndeep ensemble + conformal intervals', ha='center', fontsize=5.2, color=C_TXT)
# panel 3: intervention summary
a3 = fig.add_axes([0.745, 0.40, 0.21, 0.38])
iv = pd.read_csv(f'{R}/intervention_results.csv'); a = iv[iv.group == 'ASD']
bins = np.linspace(np.percentile(np.r_[a.random_improvement_confirmed, a.improvement_confirmed], 2), max(a.improvement_confirmed.max(), a.random_improvement_confirmed.max()), 16)
a3.hist(a.random_improvement_confirmed.clip(bins[0]), bins=bins, color='#c7c6c1', alpha=0.9, label='random search')
a3.hist(a.improvement_confirmed.clip(bins[0]), bins=bins, color='#6250d6', alpha=0.75, label='NeuronAI-guided')
a3.tick_params(labelsize=5); a3.set_yticks([])
a3.legend(fontsize=5.0, loc='upper left')
ax.text(85, 9.0, 'distance reduction to TD (%), 129 sparse\ntarget combinations, simulator-confirmed', ha='center', fontsize=5.2, color=C_TXT2)
ci = vs['child_identification']
iv_ = vs['intervention']
ax.text(50, 3.0, f"{df.child.nunique()} children · out-of-sample prediction · child identification {100*ci['accuracy']:.0f}% "
        f"(chance {100/ci['n_children']:.1f}%) · guided search better than random in "
        f"{iv_.get('ASD_children_ai_better_pct', iv_['ASD_pct_ai_better_than_random']):.0f}% of children with ASD · "
        f"pharmaco-EEG benchmarks incl. failures · open code",
        ha='center', fontsize=4.7, color=C_TXT, fontweight='bold')
fig.savefig('figures/Graphical_abstract.png', dpi=600, bbox_inches=None)
fig.savefig('figures/Graphical_abstract.pdf', bbox_inches=None)
fig.savefig('figures/Graphical_abstract.tiff', dpi=600, bbox_inches=None, pil_kwargs={'compression': 'tiff_lzw'})
print('ok')
'''

SOURCE_FILES['tests/__init__.py'] = r'''
'''

SOURCE_FILES['tests/test_neuronai.py'] = r'''"""Unit and property tests for NeuronAI v2 (run with `pytest -q`)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from neuronai.model import (simulate, theta_dict, apply_controls, CONTROLS, FIT_PARAMS, FIT_BOUNDS,
                            params_from_unit, params_to_unit, connectivity_matrices, _psp_discretisation)
from neuronai.features import extract_features, feature_matrix, phase_sync, spectral_features, psd, FEATURE_NAMES
from neuronai.stats import icc_2_1, fdr_bh, hedges_g
from neuronai.intervention import NormativeReference, all_supports
from neuronai.data import read_gdf, preprocess, _norm_name


def short_sim(seed=0, **kw):
    return simulate(theta_dict(**kw), duration=6.0, transient=1.0, seed=seed)


# ---------------------------------------------------------------- model
def test_simulation_shape_and_finite():
    r = short_sim()
    assert r.eeg.shape == (1, 17, 1500)
    assert np.isfinite(r.eeg).all()
    assert r.eeg.std() > 0


def test_bitwise_reproducibility():
    a, b = short_sim(seed=3).eeg, short_sim(seed=3).eeg
    assert np.array_equal(a, b)
    assert not np.array_equal(a, short_sim(seed=4).eeg)


def test_batch_equals_single():
    th = theta_dict()
    single = simulate(th, duration=4.0, transient=1.0, seed=1).eeg[0]
    batch = simulate({k: np.full(1, v) for k, v in th.items()}, duration=4.0, transient=1.0, seed=1).eeg[0]
    assert np.allclose(single, batch)


def test_common_average_reference():
    r = short_sim()
    assert np.allclose(r.eeg.mean(axis=1), 0, atol=1e-4)


def test_psp_discretisation_exact_steady_state():
    # step response of the unit-gain kernel converges to 1/k
    P, G = _psp_discretisation(100.0, 1e-3)
    y = np.zeros(2)
    for _ in range(2000):
        y = P @ y + G * 1.0
    assert abs(y[0] - 1 / 100.0) < 1e-9


def test_controls_zero_is_identity():
    th = {k: np.array([v]) for k, v in theta_dict().items()}
    out = apply_controls(th, np.zeros((1, len(CONTROLS))))
    for k in th:
        assert np.allclose(out[k], th[k])


@pytest.mark.parametrize('knob,param,sign', [('GABA_A', 'G', 1), ('GABA_slow', 'B', 1), ('AMPA', 'A', 1),
                                             ('Drive', 'p', 1), ('NE', 'rscale', 1), ('5HT', 'dv0', -1)])
def test_controls_direction(knob, param, sign):
    th = {k: np.array([v]) for k, v in theta_dict().items()}
    u = np.zeros((1, len(CONTROLS))); u[0, CONTROLS.index(knob)] = 1.0
    base = apply_controls(th, None)[param]
    mod = apply_controls(th, u)[param]
    assert np.sign(mod - base)[0] == sign


def test_controls_clipped():
    th = {k: np.array([v]) for k, v in theta_dict().items()}
    u = np.full((1, len(CONTROLS)), 5.0)
    assert np.allclose(apply_controls(th, u)['A'], apply_controls(th, np.ones((1, len(CONTROLS))))['A'])


def test_unit_roundtrip():
    z = np.random.default_rng(0).random((20, len(FIT_PARAMS)))
    assert np.allclose(params_to_unit(params_from_unit(z)), z)


def test_matrices_row_normalised():
    W, M = connectivity_matrices()
    assert np.allclose(W.sum(1), 1) and np.allclose(M.sum(1), 1)
    assert np.allclose(np.diag(W), 0)


# ---------------------------------------------------------------- features
def test_features_scale_invariant():
    x = short_sim().eeg[0]
    f1 = feature_matrix(extract_features(x, 250.0))
    f2 = feature_matrix(extract_features(x * 1e-6, 250.0))
    assert np.allclose(f1, f2, rtol=1e-4, atol=1e-6)


def test_sync_bounds_and_extremes():
    fs, t = 250.0, np.arange(0, 20, 1 / 250.0)
    rng = np.random.default_rng(0)
    same = np.tile(np.sin(2 * np.pi * 10 * t), (5, 1)) + 1e-3 * rng.standard_normal((5, len(t)))
    noise = rng.standard_normal((5, len(t)))
    s1, s0 = phase_sync(same, fs), phase_sync(noise, fs)
    assert s1['plv_alpha'] > 0.99 and s1['kuramoto_alpha'] > 0.99
    assert s0['plv_alpha'] < 0.2
    for s in (s0, s1):
        for k in ('plv_alpha', 'kuramoto_alpha'):
            assert 0 <= float(s[k]) <= 1


def test_wpli_zero_lag_insensitive():
    # zero-lag (volume-conduction-like) coupling must give wPLI ~ 0 while PLV ~ 1
    fs, t = 250.0, np.arange(0, 20, 1 / 250.0)
    rng = np.random.default_rng(1)
    src = np.sin(2 * np.pi * 10 * t + 0.3 * np.cumsum(rng.standard_normal(len(t))) / np.sqrt(250))
    x = np.stack([src + 0.05 * rng.standard_normal(len(t)) for _ in range(4)])
    s = phase_sync(x, fs)
    assert s['plv_alpha'] > 0.9 and abs(s['wpli_alpha']) < 0.2


def test_iaf_recovers_alpha_peak():
    fs, t = 250.0, np.arange(0, 20, 1 / 250.0)
    rng = np.random.default_rng(2)
    x = np.stack([np.sin(2 * np.pi * 10 * t + rng.random() * 6) + 0.3 * rng.standard_normal(len(t)) for _ in range(3)])
    f, P = psd(x, fs)
    assert abs(spectral_features(f, P)['iaf'] - 10.0) < 0.6


def test_exponent_of_pink_noise():
    rng = np.random.default_rng(3)
    n = 250 * 60
    spec = np.fft.rfft(rng.standard_normal((3, n)))
    fr = np.fft.rfftfreq(n, 1 / 250.0); fr[0] = 1
    x = np.fft.irfft(spec / fr ** 1.0, n)           # 1/f^2 power -> exponent 2
    f, P = psd(x, 250.0)
    assert abs(spectral_features(f, P)['exponent'] - 2.0) < 0.25


# ---------------------------------------------------------------- statistics & search
def test_icc_perfect_and_noise():
    x = np.random.default_rng(0).standard_normal(50)
    assert icc_2_1(x, x) > 0.999
    assert icc_2_1(x, np.random.default_rng(1).standard_normal(50)) < 0.4


def test_fdr_monotone():
    q = fdr_bh(np.array([0.001, 0.01, 0.03, 0.5]))
    assert np.all(np.diff(q) >= 0) and q[-1] <= 1


def test_hedges_sign():
    assert hedges_g(np.ones(10) + np.arange(10) * .1, np.arange(10) * .1) > 0


def test_supports_count():
    assert len(all_supports(3)) == 9 + 36 + 84 == 129


def test_mahalanobis_zero_at_mean():
    X = np.random.default_rng(0).standard_normal((40, len(FEATURE_NAMES)))
    ref = NormativeReference(X)
    assert ref.distance(X.mean(0)[None])[0] < 1e-8
    assert np.all(ref.distance(X) >= 0)


# ---------------------------------------------------------------- data
def test_channel_aliases():
    assert _norm_name('EEG T3-REF') == 'T7' and _norm_name('FP1') == 'Fp1'


def test_gdf_roundtrip_and_preprocess():
    import struct
    fs, n, ns = 256, 256 * 70, 17
    labels = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8']
    rng = np.random.default_rng(0)
    data = rng.standard_normal((ns, n)) * 10 + 5 * np.sin(2 * np.pi * 10 * np.arange(n) / fs)
    hdr = bytearray(256 * (ns + 1))
    hdr[:8] = b'GDF 1.25'
    hdr[184:192] = struct.pack('<q', len(hdr)); hdr[236:244] = struct.pack('<q', n)
    hdr[244:252] = struct.pack('<II', 1, fs); hdr[252:254] = struct.pack('<H', ns)
    o = 256
    def put(vals, w, fmt=None):
        nonlocal o
        for v in vals:
            hdr[o:o + w] = struct.pack(fmt, v) if fmt else v.encode().ljust(w); o += w
    put(labels, 16); put([''] * ns, 80); put(['uV'] * ns, 8)
    put([-1e4] * ns, 8, '<d'); put([1e4] * ns, 8, '<d'); put([-10000] * ns, 8, '<q'); put([10000] * ns, 8, '<q')
    put([''] * ns, 80); put([1] * ns, 4, '<i'); put([17] * ns, 4, '<I')
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 't.gdf')
        with open(p, 'wb') as f:
            f.write(bytes(hdr)); f.write(data.T.astype('<f8').tobytes())
        x, fs2, names = read_gdf(p)
        assert fs2 == fs and np.allclose(x, data) and names[0] == 'FP1'
        pp = preprocess(x, fs2, names)
        assert pp['fs'] == 250.0 and len(pp['windows']) >= 3 and not pp['excluded']


# ---------------------------------------------------------------- additions (v2.0 search & refinement)
def test_common_random_numbers():
    th = {k: np.full(4, v) for k, v in theta_dict().items()}
    r = simulate(th, duration=2.0, transient=0.5, seed=1, crn_index=np.array([0, 1, 0, 1]))
    assert np.allclose(r.eeg[0], r.eeg[2]) and not np.allclose(r.eeg[0], r.eeg[1])


def test_gaba_decay_kinetics_vectorised():
    from neuronai.model import _psp_discretisation, _psp_discretisation_vec
    P, G = _psp_discretisation(500.0, 1e-3)
    Pv, Gv = _psp_discretisation_vec(np.array([[500.0]]), 1e-3)
    assert np.allclose(P.ravel(), [x.item() for x in Pv]) and np.allclose(G, [x.item() for x in Gv])


def test_random_like_matches_support_and_norm():
    from neuronai.intervention import random_like
    U = np.zeros((5, len(CONTROLS))); U[0, [0, 3]] = [0.5, -0.4]; U[1, 2] = 0.9; U[2, [1, 4, 8]] = [0.2, 0.3, -0.1]
    U[3, 5] = -1.0; U[4, [6, 7]] = [0.7, 0.7]
    R_ = random_like(U, seed=0)
    assert np.all((np.abs(R_) > 1e-12).sum(1) == (np.abs(U) > 1e-12).sum(1))
    assert np.allclose(np.linalg.norm(R_, axis=1), np.linalg.norm(U, axis=1), atol=1e-9)
    assert np.all(np.abs(R_) <= 1)


def test_mlp_gradient_matches_finite_difference():
    from sklearn.neural_network import MLPRegressor
    from neuronai.inference import ForwardEmulator, emulator_value_grad
    rng = np.random.default_rng(0)
    Xtr = rng.random((300, 11)); Ytr = np.c_[np.sin(3 * Xtr[:, 0]), Xtr[:, 1] ** 2, Xtr[:, 2] * Xtr[:, 3]]
    emu = ForwardEmulator(n_members=2, hidden=(16, 16), max_iter=200).fit(Xtr, Ytr)
    u = rng.random(11)
    P, J = emulator_value_grad(emu, u)
    assert np.allclose(P, emu.predict(u[None])[0])
    eps = 1e-6
    for k in range(3):
        e = np.zeros(11); e[k] = eps
        num = (emu.predict((u + e)[None])[0] - emu.predict((u - e)[None])[0]) / (2 * eps)
        assert np.allclose(num, J[:, k], atol=1e-5)


def test_narrowband_line_detected_but_not_alpha():
    from neuronai.data import detect_narrowband
    fs, n = 250.0, 250 * 80
    rng = np.random.default_rng(5)
    t = np.arange(n) / fs
    spec = np.fft.rfft(rng.standard_normal((6, n)))
    fr = np.fft.rfftfreq(n, 1 / fs); fr[0] = 1
    x = np.fft.irfft(spec / fr ** 0.8, n)
    alpha = np.sum([np.sin(2 * np.pi * f * t + rng.random() * 6) for f in np.linspace(9, 11, 15)], 0) * 0.02
    line = 0.3 * np.sin(2 * np.pi * 16.0 * t)
    peaks = detect_narrowband(x + alpha + line[None] * rng.random((6, 1)), fs)
    assert any(abs(p - 16.0) < 0.2 for p in peaks)
    assert not any(8 < p < 12 for p in peaks)


def test_discriminant_identification_separable_and_null():
    from neuronai.stats import identify_discriminant
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 3, (4, 5))
    lab = np.repeat(np.arange(4), 10)
    M = centres[lab] + rng.normal(0, 0.5, (40, 5))
    assert identify_discriminant(M, lab) > 0.95
    noise = rng.normal(0, 1, (40, 5))
    assert identify_discriminant(noise, lab) < 0.5


def test_child_permutation_exact_and_monte_carlo_agree():
    from neuronai.stats import child_permutation_test
    rng = np.random.default_rng(1)
    children = np.repeat(np.arange(12), 5); groups = np.where(children < 6, 'A', 'B')
    v = rng.normal(0, 1, 60) + (groups == 'A') * 1.5 + np.repeat(rng.normal(0, 0.3, 12), 5)
    _, p_exact, n_exact = child_permutation_test(v, groups, children)
    _, p_mc, n_mc = child_permutation_test(v, groups, children, max_exact=10, n_perm=5000)
    assert n_exact == 924 and n_mc == 5000
    assert abs(p_exact - p_mc) < 0.02


def test_sign_flip_test_detects_shift_and_null():
    from neuronai.stats import sign_flip_test
    rng = np.random.default_rng(0)
    assert sign_flip_test(rng.normal(1.0, 1, 30))[1] < 0.01
    assert sign_flip_test(rng.normal(0.0, 1, 30))[1] > 0.05
'''


# ----------------------------------------------------------------------------- setup
import os, sys, subprocess, zipfile, glob, shutil, time
REPO = os.environ.get("NEURONAI_HOME", "/content/NeuronAI")
os.makedirs(REPO, exist_ok=True)
for rel, src in SOURCE_FILES.items():
    path = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
print("NeuronAI v2.0 source written to", REPO)


def run(cmd):
    print("\n$ " + cmd, flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError("Step failed: " + cmd)
    print(f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ----------------------------------------------------------------------------- data
import requests


def fetch(url, dest):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


RAW = {}
if any(v.startswith("/content/drive") for v in MANUAL_ZIPS.values()):
    from google.colab import drive
    drive.mount("/content/drive")
for grp in ["ASD", "TD"]:
    if MANUAL_ZIPS.get(grp):
        RAW[grp] = MANUAL_ZIPS[grp]
    elif DIRECT_URLS.get(grp, "").strip():
        RAW[grp] = f"/content/{grp}_all.zip"
        if not os.path.exists(RAW[grp]):
            print("downloading", grp, "...", flush=True)
            fetch(DIRECT_URLS[grp].strip(), RAW[grp])
    else:
        raise RuntimeError(f"Provide DIRECT_URLS['{grp}'] or MANUAL_ZIPS['{grp}'] (see the instructions at the top).")

# ----------------------------------------------------------------------------- pipeline
N_BANK = {"quick": 2048, "paper": 8192, "test": 256}[MODE]
os.environ["OMP_NUM_THREADS"] = "1"
if RUN_TESTS:
    run("pip -q install pytest && python -m pytest -q tests")
run(f'python scripts/00_prepare_data.py --asd "{RAW["ASD"]}" --td "{RAW["TD"]}"')
run(f"python scripts/01_generate_banks.py --kind I --n {N_BANK} & python scripts/01_generate_banks.py --kind E --n {N_BANK}; wait")
run("python scripts/02_preprocess_features.py --asd data/ASD --td data/TD")
run("python scripts/02c_assign_participants.py --meta data_meta")
run("python scripts/03_train_models.py")
run("python scripts/03b_noise_ceiling.py")
run("python scripts/03c_benchmark_inference.py")
run("python scripts/04_fit_subjects.py --shard 0 --nshards 2 & python scripts/04_fit_subjects.py --shard 1 --nshards 2; wait")
run("python scripts/05_pharmaco_validity.py & python scripts/05b_control_fingerprints.py; wait")
run("python scripts/06_intervention.py --shard 0 --nshards 2 & python scripts/06_intervention.py --shard 1 --nshards 2; wait")
run("python scripts/08_single_case.py")
run("python scripts/07_validation_stats.py")
run("python scripts/fig1_schematic.py")
run("python scripts/09_figures.py")
run("python scripts/fig6_clinical_load.py")
run("python scripts/fig_graphical_abstract.py")

# ----------------------------------------------------------------------------- show & download
try:
    from IPython.display import Image, display
    for f in sorted(glob.glob(os.path.join(REPO, "figures", "*.png"))):
        print(os.path.basename(f)); display(Image(filename=f, width=900))
except ImportError:
    print("Figures written to", os.path.join(REPO, "figures"))
print()
print("Key results:")
print(open(os.path.join(REPO, "results", "validation_summary.json")).read()[:3000])
shutil.make_archive(REPO + "_results", "zip", REPO, ".")
try:
    from google.colab import files
    files.download(REPO + "_results.zip")
except Exception as exc:
    print("Results archive:", REPO + "_results.zip")
