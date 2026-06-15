import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GeoDFN.Classes.DFNGenerator import DFNGenerator
from GeoDFN.Classes._validation import (
    VALID_APERTURE_METHODS,
    VALID_LENGTH_PDFS,
    VALID_ORIENTATION_PDFS,
    VALID_SPATIAL_PDFS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoDFN", layout="wide")
st.title("GeoDFN — Discrete Fracture Network Generator")

# ── Brazil example defaults (one per set) ─────────────────────────────────────
_D = [
    dict(I=0.0100, lpdf="Log-Normal", mu=2.40, sigma=0.73, Lmin=2.59,  Lmax=57.48,
         spdf="Power-law", sp_alpha=0.51, min_dist=1.0,
         opdf="Von-Mises", kappa=8.55,  loc_rad=1.40,  t_min=30,   t_max=120, buf=1.4),
    dict(I=0.0435, lpdf="Log-Normal", mu=2.73, sigma=0.68, Lmin=2.23,  Lmax=114.92,
         spdf="Power-law", sp_alpha=0.74, min_dist=7.5,
         opdf="Von-Mises", kappa=24.50, loc_rad=2.75,  t_min=120,  t_max=175, buf=0.8),
    dict(I=0.0256, lpdf="Log-Normal", mu=3.06, sigma=0.66, Lmin=1.20,  Lmax=121.62,
         spdf="Power-law", sp_alpha=0.80, min_dist=7.5,
         opdf="Von-Mises", kappa=58.16, loc_rad=0.063, t_min=-5,   t_max=30,  buf=1.7),
]
SET_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


# ── Sidebar: domain + aperture ────────────────────────────────────────────────
with st.sidebar:
    st.header("Domain")
    domain_x    = st.number_input("X length (m)",       value=300.0, min_value=1.0, step=10.0)
    domain_y    = st.number_input("Y length (m)",       value=600.0, min_value=1.0, step=10.0)
    num_real    = st.slider("Realizations",             1, 20, 1)
    dfn_name    = st.text_input("Run name",             value="my_DFN")
    output_dir  = st.text_input("Output folder",        value="DFNs")
    save_pic    = st.checkbox("Save plots to disk",     value=False)

    st.divider()
    st.header("Aperture")
    ap_method = st.selectbox("Method", list(VALID_APERTURE_METHODS))
    ap_params = {"method": ap_method}

    if ap_method == "subLinear":
        ap_params["scalingCoefficient"] = st.number_input("Scaling coefficient", value=0.001, format="%.5f")
        ap_params["scalingExponent"]    = st.number_input("Scaling exponent",    value=0.5,   min_value=0.0, max_value=2.0)
    elif ap_method == "constant":
        ap_params["aperture"] = st.number_input("Aperture (m)", value=1e-3, format="%.5f")
    elif ap_method == "Barton-Bandis":
        ap_params["JRC"]        = st.number_input("JRC",        value=15.0)
        ap_params["JCS"]        = st.number_input("JCS (MPa)",  value=140.0)
        ap_params["sigma_Hmax"] = st.number_input("σ_Hmax (MPa)", value=100.0)
        ap_params["sigma_c"]    = st.number_input("σ_c (MPa)",  value=130.0)
        ap_params["strike"]     = st.number_input("Strike (°)", value=95.0)
    elif ap_method == "Lepillier":
        ap_params["aperture"] = st.number_input("Initial aperture (m)", value=1e-3, format="%.5f")
        ap_params["S_Hmax"]   = st.number_input("S_Hmax (Pa)", value=1.8e8, format="%.3e")
        ap_params["S_hmin"]   = st.number_input("S_hmin (Pa)", value=0.7e8, format="%.3e")
        ap_params["E"]        = st.number_input("E (Pa)",      value=15e9,  format="%.3e")
        ap_params["nu"]       = st.number_input("ν",           value=0.22,  min_value=0.0, max_value=0.5)
        ap_params["strike"]   = st.number_input("Strike (°)",  value=95.0)

    st.divider()
    n_sets = st.slider("Number of fracture sets", 1, 5, 3)


# ── Fracture set configuration ────────────────────────────────────────────────
st.header("Fracture Sets")
max_dist = float(max(domain_x, domain_y))
set_tabs = st.tabs([f"Set {i + 1}" for i in range(n_sets)])
sets = []

for i, tab in enumerate(set_tabs):
    d = _D[i] if i < len(_D) else _D[0]
    with tab:
        c1, c2, c3 = st.columns(3)

        # ── Column 1: general ────────────────────────────────────────────────
        with c1:
            st.subheader("General")
            intensity = st.number_input("Intensity I (m⁻¹)", value=d["I"], format="%.4f", key=f"I_{i}")
            buf       = st.number_input("Buffer zone (m)",   value=d["buf"],              key=f"buf_{i}")

        # ── Column 2: length + orientation ───────────────────────────────────
        with c2:
            st.subheader("Fracture Length")
            lpdf = st.selectbox("PDF", VALID_LENGTH_PDFS,
                                index=list(VALID_LENGTH_PDFS).index(d["lpdf"]), key=f"lpdf_{i}")
            lp = {}
            if lpdf == "Log-Normal":
                lp["mu"]    = st.number_input("μ",         value=d["mu"],    key=f"mu_{i}")
                lp["sigma"] = st.number_input("σ",         value=d["sigma"], key=f"sig_{i}")
                lp["Lmin"]  = st.number_input("L_min (m)", value=d["Lmin"],  key=f"lmin_{i}")
                lp["Lmax"]  = st.number_input("L_max (m)", value=d["Lmax"],  key=f"lmax_{i}")
            elif lpdf == "Power-law":
                lp["alpha"] = st.number_input("α",         value=2.0,        key=f"alpha_l_{i}")
                lp["Lmin"]  = st.number_input("L_min (m)", value=d["Lmin"],  key=f"lmin_{i}")
                lp["Lmax"]  = st.number_input("L_max (m)", value=d["Lmax"],  key=f"lmax_{i}")
            elif lpdf == "Constant":
                lp["L"]     = st.number_input("L (m)",     value=10.0,       key=f"L_{i}")
            elif lpdf == "Exponential":
                lp["lambda"] = st.number_input("λ",        value=0.1,        key=f"lam_{i}")
                lp["Lmin"]   = st.number_input("L_min (m)",value=d["Lmin"],  key=f"lmin_{i}")
                lp["Lmax"]   = st.number_input("L_max (m)",value=d["Lmax"],  key=f"lmax_{i}")

            st.subheader("Orientation")
            opdf = st.selectbox("PDF", VALID_ORIENTATION_PDFS, key=f"opdf_{i}")
            op = {}
            if opdf == "Von-Mises":
                loc_deg     = st.number_input("Mean (°)",         value=float(np.degrees(d["loc_rad"])), key=f"loc_{i}")
                op["loc"]   = np.radians(loc_deg)
                op["kappa"] = st.number_input("κ (concentration)", value=d["kappa"], key=f"kappa_{i}")
                t_min_deg   = st.number_input("θ_min (°)",        value=float(d["t_min"]), key=f"tmin_{i}")
                t_max_deg   = st.number_input("θ_max (°)",        value=float(d["t_max"]), key=f"tmax_{i}")
                op["thetaMin"] = np.radians(t_min_deg)
                op["thetaMax"] = np.radians(t_max_deg)
            elif opdf == "Uniform":
                t_min_deg   = st.number_input("θ_min (°)", value=0.0,   key=f"tmin_{i}")
                t_max_deg   = st.number_input("θ_max (°)", value=180.0, key=f"tmax_{i}")
                op["thetaMin"] = np.radians(t_min_deg)
                op["thetaMax"] = np.radians(t_max_deg)
            elif opdf == "Constant":
                theta_deg  = st.number_input("θ (°)", value=90.0, key=f"theta_{i}")
                op["theta"] = np.radians(theta_deg)

        # ── Column 3: spatial distribution ───────────────────────────────────
        with c3:
            st.subheader("Spatial Distribution")
            spdf = st.selectbox("PDF", VALID_SPATIAL_PDFS,
                                index=list(VALID_SPATIAL_PDFS).index(d["spdf"]), key=f"spdf_{i}")
            sp = {"max distance": max_dist}
            st.caption(f"Max distance set to {max_dist:.0f} m (= max domain side)")
            if spdf == "Power-law":
                sp["alpha"]        = st.number_input("α",              value=d["sp_alpha"], key=f"sp_alpha_{i}")
                sp["min distance"] = st.number_input("Min distance (m)", value=d["min_dist"], key=f"mindist_{i}")
            elif spdf == "Log-Normal":
                sp["mu"]    = st.number_input("μ", value=2.0, key=f"smu_{i}")
                sp["sigma"] = st.number_input("σ", value=0.5, key=f"ssig_{i}")

        sets.append({
            "I": intensity,
            "fractureLengthPDF": lpdf,
            "fractureLengthPDFParams": lp,
            "spatialDistributionPDF": spdf,
            "spatialDistributionPDFParams": sp,
            "orientationDistributionPDF": opdf,
            "orientationDistributionPDFParams": op,
            "bufferZone": {"method": "constant", "constant": buf},
        })


# ── Generate button ───────────────────────────────────────────────────────────
st.divider()
if st.button("Generate DFN", type="primary", use_container_width=False):
    try:
        with st.spinner(f"Generating {num_real} realization(s) — this may take a minute…"):
            gen = DFNGenerator(
                domain_x, domain_y, sets, ap_params, dfn_name,
                numOfRealizations=num_real, savePic=save_pic, output_dir=output_dir,
            )
        st.session_state["gen"]    = gen
        st.session_state["domain"] = (domain_x, domain_y)
        if len(gen.realizations) < num_real:
            st.warning(
                f"{len(gen.realizations)} of {num_real} realization(s) converged "
                f"(others hit the placement retry limit)."
            )
        else:
            st.success(f"{len(gen.realizations)} realization(s) generated successfully.")
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
        real_tabs = st.tabs([f"Realization {i + 1}" for i in range(len(gen.realizations))])

        for r_idx, (tab, realization) in enumerate(zip(real_tabs, gen.realizations)):
            with tab:
                col_plot, col_stats = st.columns([2, 1])

                with col_plot:
                    fig, ax = plt.subplots(figsize=(5, 9))
                    ax.set_xlim(0, dx)
                    ax.set_ylim(0, dy)
                    ax.set_aspect("equal")
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    ax.set_facecolor("#f5f5f5")
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
                        patches.append(mpatches.Patch(color=color, label=f"Set {s_idx + 1} ({len(frac_set)})"))

                    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)
                    st.pyplot(fig)
                    plt.close(fig)

                with col_stats:
                    st.subheader("Statistics")
                    all_fracs = [f for fset in realization for f in fset]
                    lengths   = [f["fracture length"] for f in all_fracs]
                    apertures = [f["fracture aperture"] for f in all_fracs]
                    total_intensity = sum(lengths) / (dx * dy)

                    st.metric("Total fractures",      len(all_fracs))
                    st.metric("Total intensity (m⁻¹)", f"{total_intensity:.5f}")
                    st.metric("Mean length (m)",       f"{np.mean(lengths):.2f}")
                    st.metric("Max length (m)",        f"{np.max(lengths):.2f}")
                    st.metric("Mean aperture (m)",     f"{np.mean(apertures):.2e}")

                    st.divider()
                    st.caption("Per set")
                    for s_idx, fset in enumerate(realization):
                        n = len(fset)
                        color = SET_COLORS[s_idx % len(SET_COLORS)]
                        st.markdown(
                            f'<span style="color:{color}">■</span> **Set {s_idx + 1}:** {n} fractures',
                            unsafe_allow_html=True,
                        )
