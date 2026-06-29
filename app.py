import io
import os
import sys
import zipfile

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GeoDFN.Classes.DFNGenerator import DFNGenerator
from GeoDFN.Classes._validation import (
    VALID_APERTURE_METHODS, VALID_LENGTH_PDFS, VALID_ORIENTATION_PDFS, VALID_SPATIAL_PDFS,
)

# ── Logo ──────────────────────────────────────────────────────────────────────
_base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
_logo_path = os.path.join(_base, 'logoGeoDFN.png')
_logo = Image.open(_logo_path) if os.path.exists(_logo_path) else None

st.set_page_config(page_title="GeoDFN", page_icon=_logo, layout="wide")

st.markdown("""
<style>
[data-testid="stSelectbox"],
[data-testid="stNumberInput"],
[data-testid="stTextInput"] {
    max-width: 280px !important;
    width: 280px !important;
}
</style>
""", unsafe_allow_html=True)

_L_PDFS  = list(VALID_LENGTH_PDFS)
_O_PDFS  = list(VALID_ORIENTATION_PDFS)
_S_PDFS  = list(VALID_SPATIAL_PDFS)
_AP_MTHS = list(VALID_APERTURE_METHODS)
SET_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
_UNSET = "— select —"

# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "Custom": {
        "domain_x": 100.0, "domain_y": 100.0, "n_sets": 1,
        "ap_method": _UNSET,
        "sets": [
            dict(I=0.0, buf=0.0,
                 lpdf=_UNSET, opdf=_UNSET, spdf=_UNSET),
        ],
    },
    "Brazil Apodi": {
        "domain_x": 300.0, "domain_y": 600.0, "n_sets": 3,
        "ap_method": "subLinear", "ap_coeff": 0.001, "ap_exp": 0.5,
        "sets": [
            dict(I=0.0100, buf=1.4,
                 lpdf="Log-Normal", mu=2.40, sig=0.73, lmin=2.59, lmax=57.48,
                 opdf="Von-Mises", loc_deg=80.2, kappa=8.55, tmin_deg=30.0, tmax_deg=120.0,
                 spdf="Power-law", sp_alpha=0.51, mindist=1.0),
            dict(I=0.0435, buf=0.8,
                 lpdf="Log-Normal", mu=2.73, sig=0.68, lmin=2.23, lmax=114.92,
                 opdf="Von-Mises", loc_deg=157.5, kappa=24.50, tmin_deg=120.0, tmax_deg=175.0,
                 spdf="Power-law", sp_alpha=0.74, mindist=7.5),
            dict(I=0.0256, buf=1.7,
                 lpdf="Log-Normal", mu=3.06, sig=0.66, lmin=1.20, lmax=121.62,
                 opdf="Von-Mises", loc_deg=3.6, kappa=58.16, tmin_deg=-5.0, tmax_deg=30.0,
                 spdf="Power-law", sp_alpha=0.80, mindist=7.5),
        ],
    },
    "Single Set": {
        "domain_x": 200.0, "domain_y": 200.0, "n_sets": 1,
        "ap_method": "constant", "ap_aperture": 0.001,
        "sets": [
            dict(I=0.02, buf=1.0,
                 lpdf="Constant", L=15.0,
                 opdf="Uniform", tmin_deg=60.0, tmax_deg=120.0,
                 spdf="Uniform"),
        ],
    },
    "Orthogonal set": {
        "domain_x": 100.0, "domain_y": 100.0, "n_sets": 2,
        "ap_method": "subLinear", "ap_coeff": 0.001, "ap_exp": 0.5,
        "sets": [
            dict(I=0.03, buf=1.0,
                 lpdf="Log-Normal", mu=2.3, sig=0.5, lmin=3.0, lmax=40.0,
                 opdf="Von-Mises", loc_deg=0, kappa=80.0, tmin_deg=-10.0, tmax_deg=10.0,
                 spdf="Power-law", sp_alpha=0.6, mindist=2.0),
            dict(I=0.025, buf=1.0,
                 lpdf="Log-Normal", mu=1.8, sig=0.5, lmin=2.0, lmax=25.0,
                 opdf="Von-Mises", loc_deg=90.0, kappa=80.0, tmin_deg=80.0, tmax_deg=100.0,
                 spdf="Power-law", sp_alpha=0.6, mindist=2.0),
        ],
    },
    "Conjugate set": {
        "domain_x": 200.0, "domain_y": 200.0, "n_sets": 2,
        "ap_method": "subLinear", "ap_coeff": 0.0008, "ap_exp": 0.6,
        "sets": [
            dict(I=0.018, buf=1.2,
                 lpdf="Log-Normal", mu=2.5, sig=0.6, lmin=3.0, lmax=50.0,
                 opdf="Von-Mises", loc_deg=60.0, kappa=12.0, tmin_deg=40.0, tmax_deg=80.0,
                 spdf="Power-law", sp_alpha=0.55, mindist=2.5),
            dict(I=0.018, buf=1.2,
                 lpdf="Log-Normal", mu=2.5, sig=0.6, lmin=3.0, lmax=50.0,
                 opdf="Von-Mises", loc_deg=120.0, kappa=12.0, tmin_deg=100.0, tmax_deg=140.0,
                 spdf="Power-law", sp_alpha=0.55, mindist=2.5),
        ],
    },
}
PRESET_NAMES = list(PRESETS.keys())


def _apply_preset():
    name = st.session_state.get("preset_select", PRESET_NAMES[0])
    p = PRESETS[name]
    st.session_state["domain_x"]    = p["domain_x"]
    st.session_state["domain_y"]    = p["domain_y"]
    st.session_state["n_sets"]      = p["n_sets"]
    st.session_state["ap_method"]   = p["ap_method"]
    st.session_state["ap_coeff"]    = p.get("ap_coeff", 0.001)
    st.session_state["ap_exp"]      = p.get("ap_exp", 0.5)
    st.session_state["ap_aperture"] = p.get("ap_aperture", 0.001)
    for i, s in enumerate(p["sets"]):
        st.session_state[f"I_{i}"]          = float(s["I"])
        st.session_state[f"buf_{i}"]        = float(s["buf"])
        st.session_state[f"buf_method_{i}"] = s.get("buf_method", "constant")
        st.session_state[f"lpdf_{i}"] = s["lpdf"]
        st.session_state[f"opdf_{i}"] = s["opdf"]
        st.session_state[f"spdf_{i}"] = s["spdf"]
        if s["lpdf"] == "Log-Normal":
            st.session_state[f"mu_{i}"]   = float(s.get("mu",   2.0))
            st.session_state[f"sig_{i}"]  = float(s.get("sig",  0.5))
            st.session_state[f"lmin_{i}"] = float(s.get("lmin", 1.0))
            st.session_state[f"lmax_{i}"] = float(s.get("lmax", 50.0))
        elif s["lpdf"] == "Constant":
            st.session_state[f"L_{i}"]    = float(s.get("L", 10.0))
        elif s["lpdf"] == "Power-law":
            st.session_state[f"alpha_l_{i}"] = float(s.get("alpha_l", 2.0))
            st.session_state[f"lmin_{i}"]    = float(s.get("lmin", 1.0))
            st.session_state[f"lmax_{i}"]    = float(s.get("lmax", 50.0))
        elif s["lpdf"] == "Exponential":
            st.session_state[f"lam_{i}"]  = float(s.get("lam",  0.1))
            st.session_state[f"lmin_{i}"] = float(s.get("lmin", 1.0))
            st.session_state[f"lmax_{i}"] = float(s.get("lmax", 50.0))
        if s["opdf"] == "Von-Mises":
            st.session_state[f"loc_{i}"]   = float(s.get("loc_deg",  90.0))
            st.session_state[f"kappa_{i}"] = float(s.get("kappa",    10.0))
            st.session_state[f"tmin_{i}"]  = float(s.get("tmin_deg", 60.0))
            st.session_state[f"tmax_{i}"]  = float(s.get("tmax_deg", 120.0))
        elif s["opdf"] == "Uniform":
            st.session_state[f"tmin_{i}"]  = float(s.get("tmin_deg", 0.0))
            st.session_state[f"tmax_{i}"]  = float(s.get("tmax_deg", 180.0))
        elif s["opdf"] == "Constant":
            st.session_state[f"theta_{i}"] = float(s.get("theta_deg", 90.0))
        if s["spdf"] == "Power-law":
            st.session_state[f"sp_alpha_{i}"] = float(s.get("sp_alpha", 0.5))
            st.session_state[f"mindist_{i}"]  = float(s.get("mindist",  2.0))
        elif s["spdf"] == "Log-Normal":
            st.session_state[f"smu_{i}"]  = float(s.get("smu",  2.0))
            st.session_state[f"ssig_{i}"] = float(s.get("ssig", 0.5))


# ── Initialize state on very first run ────────────────────────────────────────
if "preset_select" not in st.session_state:
    st.session_state["num_real"]       = 1
    st.session_state["dfn_name"]       = "my_DFN"
    st.session_state["output_dir_val"] = "DFNs"
    for _j in range(3, 5):
        st.session_state[f"I_{_j}"]        = 0.01
        st.session_state[f"buf_{_j}"]        = 1.0
        st.session_state[f"buf_method_{_j}"] = "constant"
        st.session_state[f"lpdf_{_j}"]     = "Log-Normal"
        st.session_state[f"mu_{_j}"]       = 2.0
        st.session_state[f"sig_{_j}"]      = 0.5
        st.session_state[f"lmin_{_j}"]     = 1.0
        st.session_state[f"lmax_{_j}"]     = 50.0
        st.session_state[f"opdf_{_j}"]     = "Von-Mises"
        st.session_state[f"loc_{_j}"]      = 45.0
        st.session_state[f"kappa_{_j}"]    = 10.0
        st.session_state[f"tmin_{_j}"]     = 30.0
        st.session_state[f"tmax_{_j}"]     = 60.0
        st.session_state[f"spdf_{_j}"]     = "Power-law"
        st.session_state[f"sp_alpha_{_j}"] = 0.5
        st.session_state[f"mindist_{_j}"]  = 2.0
    st.session_state["preset_select"] = PRESET_NAMES[0]
    _apply_preset()


# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if _logo:
        st.image(_logo, width=200)
with col_title:
    st.title("Geologically Consistent Discrete Fracture Network Generator")

st.divider()

# _row is used in the fracture-set section below
def _row(label, widget_fn, lw=1, iw=2):
    lc, ic = st.columns([lw, iw])
    lc.markdown(f"<div style='line-height:2.4rem'>{label}</div>", unsafe_allow_html=True)
    with ic:
        return widget_fn()

# Shared grid used everywhere: [1,2,1,2,1,2] — 3 label+input pairs per row.
_G = [1, 2, 1, 2, 1, 2]

def _lbl(col, text):
    col.markdown(f"<div style='line-height:2.4rem'>{text}</div>", unsafe_allow_html=True)

# ── Row 1: Preset ─────────────────────────────────────────────────────────────
c = st.columns(_G)
_lbl(c[0], "Preset")
with c[1]:
    st.selectbox("Geological scenario", PRESET_NAMES, key="preset_select",
                 on_change=_apply_preset, label_visibility="collapsed",
                 help="Load a pre-configured geological scenario — all parameters update automatically.")

# ── Row 2: Run settings ───────────────────────────────────────────────────────
c = st.columns(_G)
_lbl(c[0], "Number of Realizations")
with c[1]:
    num_real = st.number_input("Realizations", min_value=1, max_value=1000, step=1, key="num_real", label_visibility="collapsed")
_lbl(c[2], "Run name")
with c[3]:
    dfn_name = st.text_input("Run name", key="dfn_name", label_visibility="collapsed")
_lbl(c[4], "Output folder")
with c[5]:
    output_dir = st.text_input("Output folder", key="output_dir_val", label_visibility="collapsed")

st.divider()

# ── Row 3: Domain ─────────────────────────────────────────────────────────────
c = st.columns(_G)
_lbl(c[0], "X length (m)")
with c[1]:
    domain_x = st.number_input("X length (m)", min_value=1.0, step=10.0, key="domain_x", label_visibility="collapsed")
_lbl(c[2], "Y length (m)")
with c[3]:
    domain_y = st.number_input("Y length (m)", min_value=1.0, step=10.0, key="domain_y", label_visibility="collapsed")
_lbl(c[4], "Number of Fracture sets")
with c[5]:
    n_sets = st.number_input("Fracture sets", min_value=1, max_value=5, step=1, key="n_sets", label_visibility="collapsed")

st.divider()
# ── Aperture ──────────────────────────────────────────────────────────────────
st.header("Fracture Aperture")

def _ap_row(label, widget_fn):
    c = st.columns(_G)
    _lbl(c[0], label)
    with c[1]:
        return widget_fn()

ap_method = _ap_row("Method", lambda: st.selectbox("Method", [_UNSET] + _AP_MTHS, key="ap_method", label_visibility="collapsed"))
ap_params = {"method": ap_method}

if ap_method == "subLinear":
    ap_params["scalingCoefficient"] = _ap_row("Scaling coeff.", lambda: st.number_input("Scaling coeff.", format="%.5f", key="ap_coeff", label_visibility="collapsed"))
    ap_params["scalingExponent"]    = _ap_row("Scaling exp.",   lambda: st.number_input("Scaling exp.", min_value=0.0, max_value=2.0, key="ap_exp", label_visibility="collapsed"))

elif ap_method == "constant":
    ap_params["aperture"] = _ap_row("Aperture (m)", lambda: st.number_input("Aperture (m)", format="%.5f", key="ap_aperture", label_visibility="collapsed"))

elif ap_method == "Barton-Bandis":
    ap_params["JRC"]        = _ap_row("JRC",          lambda: st.number_input("JRC",          value=15.0,  key="ap_jrc",       label_visibility="collapsed"))
    ap_params["JCS"]        = _ap_row("JCS (MPa)",    lambda: st.number_input("JCS (MPa)",    value=140.0, key="ap_jcs",       label_visibility="collapsed"))
    ap_params["sigma_Hmax"] = _ap_row("σ_Hmax (MPa)", lambda: st.number_input("σ_Hmax (MPa)", value=100.0, key="ap_shmax",     label_visibility="collapsed"))
    ap_params["sigma_c"]    = _ap_row("σ_c (MPa)",    lambda: st.number_input("σ_c (MPa)",    value=130.0, key="ap_sc",        label_visibility="collapsed"))
    ap_params["strike"]     = _ap_row("Strike (°)",   lambda: st.number_input("Strike (°)",   value=95.0,  key="ap_strike_bb", label_visibility="collapsed"))

elif ap_method == "Lepillier":
    ap_params["aperture"] = _ap_row("Initial aperture (m)", lambda: st.number_input("Initial aperture (m)", value=1e-3, format="%.5f", key="ap_lep_ap",  label_visibility="collapsed"))
    ap_params["S_Hmax"]   = _ap_row("S_Hmax (Pa)",          lambda: st.number_input("S_Hmax (Pa)",  value=1.8e8, format="%.3e", key="ap_shmax_l", label_visibility="collapsed"))
    ap_params["S_hmin"]   = _ap_row("S_hmin (Pa)",          lambda: st.number_input("S_hmin (Pa)",  value=0.7e8, format="%.3e", key="ap_shmin_l", label_visibility="collapsed"))
    ap_params["E"]        = _ap_row("E (Pa)",                lambda: st.number_input("E (Pa)",        value=15e9,  format="%.3e", key="ap_E_l",    label_visibility="collapsed"))
    ap_params["nu"]       = _ap_row("ν",                     lambda: st.number_input("ν",             value=0.22,  min_value=0.0, max_value=0.5, key="ap_nu_l", label_visibility="collapsed"))
    ap_params["strike"]   = _ap_row("Strike (°)",            lambda: st.number_input("Strike (°)",   value=95.0,  key="ap_strike_l", label_visibility="collapsed"))

st.divider()

# ── Fracture set tabs ─────────────────────────────────────────────────────────
st.header("Fracture Sets")
max_dist = float(max(domain_x, domain_y))
set_tabs = st.tabs([f"Set {i + 1}" for i in range(n_sets)])
sets = []

for i, tab in enumerate(set_tabs):
    with tab:
        # Row 1 — General (same grid as top panel)
        g = st.columns(_G)
        _lbl(g[0], "Intensity I (m⁻¹)")
        with g[1]:
            intensity = st.number_input("Intensity I (m⁻¹)", format="%.4f", key=f"I_{i}", label_visibility="collapsed")
        _lbl(g[2], "Buffer zone method")
        with g[3]:
            buf_method = st.selectbox(
                "Buffer zone method",
                ["constant", "linearRelationshipLength"],
                key=f"buf_method_{i}",
                format_func=lambda m: "Constant (m)" if m == "constant" else "Linear (× length)",
                label_visibility="collapsed",
            )
        _lbl(g[4], "Constant (m)" if buf_method == "constant" else "Coefficient")
        with g[5]:
            buf = st.number_input("Buffer value", key=f"buf_{i}", min_value=0.0, format="%.4f", label_visibility="collapsed")

        st.divider()

        # Row 2 — distributions, same _G grid
        # Section headers aligned to the same grid
        h0, h1, h2 = st.columns([3, 3, 3])
        h0.markdown("**Fracture Length**")
        h1.markdown("**Spatial Distribution**")
        h2.markdown("**Orientation**")

        # Distribution selector row
        c = st.columns(_G)
        _lbl(c[0], "Distribution")
        with c[1]: lpdf = st.selectbox("lpdf", [_UNSET] + _L_PDFS, key=f"lpdf_{i}", label_visibility="collapsed")
        _lbl(c[2], "Distribution")
        with c[3]: spdf = st.selectbox("spdf", [_UNSET] + _S_PDFS, key=f"spdf_{i}", label_visibility="collapsed")
        if spdf != _UNSET:
            st.caption(f"Max distance = {max_dist:.0f} m", help="Used as upper bound for spatial distribution")
        _lbl(c[4], "Distribution")
        with c[5]: opdf = st.selectbox("opdf", [_UNSET] + _O_PDFS, key=f"opdf_{i}", label_visibility="collapsed")

        lp, sp, op = {}, {"max distance": max_dist}, {}
        loc_deg = tmin_deg = tmax_deg = theta_deg = 0.0

        # Number of param rows needed per distribution
        _nL = {"Log-Normal": 4, "Power-law": 3, "Constant": 1, "Exponential": 3}.get(lpdf, 0)
        _nS = {"Power-law": 2, "Log-Normal": 2}.get(spdf, 0)
        _nO = {"Von-Mises": 4, "Uniform": 2, "Constant": 1}.get(opdf, 0)

        for row in range(max(_nL, _nS, _nO)):
            c = st.columns(_G)

            # Fracture Length
            if lpdf == "Log-Normal":
                if row == 0:   _lbl(c[0], "μ");         lp["mu"]    = c[1].number_input("μ",    key=f"mu_{i}",   label_visibility="collapsed")
                elif row == 1: _lbl(c[0], "σ");         lp["sigma"] = c[1].number_input("σ",    key=f"sig_{i}",  label_visibility="collapsed")
                elif row==2: _lbl(c[0], "L_min (m)"); lp["Lmin"]   = c[1].number_input("Lmin", key=f"lmin_{i}",  label_visibility="collapsed")
                elif row==3: _lbl(c[0], "L_max (m)"); lp["Lmax"]   = c[1].number_input("Lmax", key=f"lmax_{i}",  label_visibility="collapsed")
            elif lpdf == "Power-law":
                if row == 0: _lbl(c[0], "α");         lp["alpha"]  = c[1].number_input("α",    key=f"alpha_l_{i}", label_visibility="collapsed")
                elif row==1: _lbl(c[0], "L_min (m)"); lp["Lmin"]   = c[1].number_input("Lmin", key=f"lmin_{i}",    label_visibility="collapsed")
                elif row==2: _lbl(c[0], "L_max (m)"); lp["Lmax"]   = c[1].number_input("Lmax", key=f"lmax_{i}",    label_visibility="collapsed")
            elif lpdf == "Constant":
                if row == 0:
                    _lbl(c[0], "L (m)")
                    lp["L"] = c[1].number_input("L", key=f"L_{i}", label_visibility="collapsed")
                    lp["Lmin"] = lp["L"]
                    lp["Lmax"] = lp["L"]
            elif lpdf == "Exponential":
                if row == 0: _lbl(c[0], "λ");         lp["lambda"] = c[1].number_input("λ",    key=f"lam_{i}",     label_visibility="collapsed")
                elif row==1: _lbl(c[0], "L_min (m)"); lp["Lmin"]   = c[1].number_input("Lmin", key=f"lmin_{i}",    label_visibility="collapsed")
                elif row==2: _lbl(c[0], "L_max (m)"); lp["Lmax"]   = c[1].number_input("Lmax", key=f"lmax_{i}",    label_visibility="collapsed")

            # Spatial Distribution
            if spdf == "Power-law":
                if row == 0: _lbl(c[2], "α");              sp["alpha"]        = c[3].number_input("α",        key=f"sp_alpha_{i}", label_visibility="collapsed")
                elif row==1: _lbl(c[2], "Min dist. (m)");  sp["min distance"] = c[3].number_input("Min dist", key=f"mindist_{i}",  label_visibility="collapsed")
            elif spdf == "Log-Normal":
                if row == 0: _lbl(c[2], "μ"); sp["mu"]    = c[3].number_input("μ", key=f"smu_{i}",  label_visibility="collapsed")
                elif row==1: _lbl(c[2], "σ"); sp["sigma"] = c[3].number_input("σ", key=f"ssig_{i}", label_visibility="collapsed")

            # Orientation
            if opdf == "Von-Mises":
                if row == 0: _lbl(c[4], "Mean (°)");   loc_deg      = c[5].number_input("Mean",  key=f"loc_{i}",   label_visibility="collapsed")
                elif row==1: _lbl(c[4], "κ");          op["kappa"]  = c[5].number_input("κ",     key=f"kappa_{i}", label_visibility="collapsed")
                elif row==2: _lbl(c[4], "θ_min (°)"); tmin_deg     = c[5].number_input("tmin",  key=f"tmin_{i}",  label_visibility="collapsed")
                elif row==3: _lbl(c[4], "θ_max (°)"); tmax_deg     = c[5].number_input("tmax",  key=f"tmax_{i}",  label_visibility="collapsed")
            elif opdf == "Uniform":
                if row == 0: _lbl(c[4], "θ_min (°)"); tmin_deg = c[5].number_input("tmin", key=f"tmin_{i}", label_visibility="collapsed")
                elif row==1: _lbl(c[4], "θ_max (°)"); tmax_deg = c[5].number_input("tmax", key=f"tmax_{i}", label_visibility="collapsed")
            elif opdf == "Constant":
                if row == 0: _lbl(c[4], "θ (°)"); theta_deg = c[5].number_input("θ", key=f"theta_{i}", label_visibility="collapsed")

        # Finalise orientation params
        if opdf == "Von-Mises":
            op["loc"] = np.radians(loc_deg)
            op["thetaMin"] = np.radians(tmin_deg)
            op["thetaMax"] = np.radians(tmax_deg)
        elif opdf == "Uniform":
            op["thetaMin"] = np.radians(tmin_deg)
            op["thetaMax"] = np.radians(tmax_deg)
        elif opdf == "Constant":
            op["theta"] = np.radians(theta_deg)

        sets.append({
            "I": intensity,
            "fractureLengthPDF": lpdf,
            "fractureLengthPDFParams": lp,
            "spatialDistributionPDF": spdf,
            "spatialDistributionPDFParams": sp,
            "orientationDistributionPDF": opdf,
            "orientationDistributionPDFParams": op,
            "bufferZone": {"method": buf_method, "constant": buf},
        })


# ── Generate ──────────────────────────────────────────────────────────────────
st.divider()
if st.button("Generate DFN", type="primary"):
    try:
        _bar    = st.progress(0, text="Starting generation…")
        _status = st.empty()

        def _on_progress(done, total):
            _bar.progress(done / total,
                          text=f"Realization {done} of {total} complete…")

        gen = DFNGenerator(
            domain_x, domain_y, sets, ap_params, dfn_name,
            numOfRealizations=num_real, savePic=True,
            output_dir=output_dir, progress_callback=_on_progress,
        )
        _bar.progress(1.0, text="Done!")
        st.session_state["gen"]    = gen
        st.session_state["domain"] = (domain_x, domain_y)
        n_ok = len(gen.realizations)
        if n_ok < num_real:
            st.warning(
                f"{n_ok} of {num_real} realization(s) converged. "
                "Others hit the placement retry limit — try reducing intensity or buffer zone."
            )
        else:
            st.success(f"{n_ok} realization(s) generated successfully.")
    except ValueError as e:
        st.error(f"Invalid parameters: {e}")


# ── Results ───────────────────────────────────────────────────────────────────
if "gen" in st.session_state:
    gen = st.session_state["gen"]
    dx, dy = st.session_state["domain"]

    if not gen.realizations:
        st.error("No realizations converged. Try reducing fracture intensity or buffer zone size.")
    else:
        st.header("Results")

        if os.path.isdir(gen.outputDir):
            _zip_buf = io.BytesIO()
            with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                for _root, _dirs, _files in os.walk(gen.outputDir):
                    for _fname in _files:
                        _fp = os.path.join(_root, _fname)
                        _zf.write(_fp, os.path.relpath(_fp, os.path.dirname(gen.outputDir)))
            _zip_buf.seek(0)
            st.download_button(
                "⬇ Download Results (ZIP)",
                data=_zip_buf,
                file_name=f"{os.path.basename(gen.outputDir)}.zip",
                mime="application/zip",
            )

        real_tabs = st.tabs([f"Realization {i + 1}" for i in range(len(gen.realizations))])
        for r_idx, (tab, realization) in enumerate(zip(real_tabs, gen.realizations)):
            with tab:
                col_plot, col_stats = st.columns([2, 1])

                with col_plot:
                    fig, ax = plt.subplots(figsize=(5, 9))
                    ax.set_xlim(0, dx)
                    ax.set_ylim(0, dy)
                    ax.set_aspect("equal")
                    ax.set_facecolor("#f5f5f5")
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    ax.set_title(f"Realization {r_idx + 1}")
                    patches = []
                    for s_idx, frac_set in enumerate(realization):
                        color = SET_COLORS[s_idx % len(SET_COLORS)]
                        for frac in frac_set:
                            ax.plot(
                                [frac["x_start"], frac["x_end"]],
                                [frac["y_start"], frac["y_end"]],
                                color=color, linewidth=0.7, alpha=0.85,
                            )
                        patches.append(
                            mpatches.Patch(color=color, label=f"Set {s_idx + 1}")
                        )
                    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)
                    st.pyplot(fig)
                    plt.close(fig)

                with col_stats:
                    # Orientation rose diagram
                    _n_bins = 18  # 10° bins
                    _bin_edges = np.linspace(0, np.pi, _n_bins + 1)
                    _fig_r, _ax_r = plt.subplots(figsize=(3, 3), subplot_kw=dict(projection="polar"))
                    for s_idx, frac_set in enumerate(realization):
                        _angs = []
                        for frac in frac_set:
                            _a = np.arctan2(frac["y_end"] - frac["y_start"],
                                            frac["x_end"] - frac["x_start"]) % np.pi
                            _angs.append(_a)
                        _counts, _ = np.histogram(_angs, bins=_bin_edges)
                        _w = np.pi / _n_bins
                        _color = SET_COLORS[s_idx % len(SET_COLORS)]
                        _ax_r.bar(_bin_edges[:-1],           _counts, width=_w, color=_color, alpha=0.6, edgecolor="none")
                        _ax_r.bar(_bin_edges[:-1] + np.pi,  _counts, width=_w, color=_color, alpha=0.6, edgecolor="none")
                    _ax_r.set_theta_zero_location("N")
                    _ax_r.set_theta_direction(-1)
                    _ax_r.set_yticklabels([])
                    _ax_r.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
                    _ax_r.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=7)
                    _ax_r.set_title("Orientation", fontsize=9, pad=8)
                    st.pyplot(_fig_r)
                    plt.close(_fig_r)

                    st.subheader("Statistics")
                    all_fracs = [f for fset in realization for f in fset]
                    lengths   = [f["fracture length"]   for f in all_fracs]
                    apertures = [f["fracture aperture"] for f in all_fracs]
                    total_intensity = sum(lengths) / (dx * dy)
                    st.metric("Total fractures",        len(all_fracs))
                    st.metric("Total intensity (m⁻¹)",  f"{total_intensity:.5f}")
                    st.metric("Mean length (m)",         f"{np.mean(lengths):.2f}")
                    st.metric("Max length (m)",          f"{np.max(lengths):.2f}")
                    st.metric("Mean aperture (m)",       f"{np.mean(apertures):.2e}")
                    st.divider()
                    st.caption("Per set")
                    for s_idx, fset in enumerate(realization):
                        color = SET_COLORS[s_idx % len(SET_COLORS)]
                        st.markdown(
                            f'<span style="color:{color}">■</span> **Set {s_idx + 1}:** {len(fset)} fractures',
                            unsafe_allow_html=True,
                        )
