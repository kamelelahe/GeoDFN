# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

# Collect everything Streamlit needs (static assets, runtime files, etc.)
st_datas, st_binaries, st_hidden = collect_all("streamlit")

# Also collect altair (Streamlit uses it for charts)
altair_datas, altair_binaries, altair_hidden = collect_all("altair")

a = Analysis(
    ["launcher.py"],
    pathex=[os.path.abspath(".")],
    binaries=st_binaries + altair_binaries,
    datas=st_datas + altair_datas + [
        ("app.py",          "."),
        ("GeoDFN",          "GeoDFN"),
        ("logoGeoDFN.png",  "."),
    ],
    hiddenimports=st_hidden + altair_hidden + [
        "streamlit.web.cli",
        "streamlit.web.server",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner",
        "streamlit.runtime.metrics_util",
        "numpy",
        "scipy",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["numba"],   # numba was removed in v2 refactor
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GeoDFN",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # no terminal window
    icon="logoGeoDFN.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="GeoDFN",
)
