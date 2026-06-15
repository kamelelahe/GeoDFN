"""
Launcher for the GeoDFN desktop app.
Starts the Streamlit server and opens the browser automatically.
"""
import os
import sys
import threading
import webbrowser
import time


def _open_browser():
    time.sleep(4)
    webbrowser.open("http://localhost:8501")


if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

app_path = os.path.join(bundle_dir, "app.py")

threading.Thread(target=_open_browser, daemon=True).start()

sys.argv = [
    "streamlit", "run", app_path,
    "--global.developmentMode=false",
    "--server.headless=true",
    "--server.port=8501",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false",
    "--browser.gatherUsageStats=false",
]

from streamlit.web import cli as stcli
stcli.main()
