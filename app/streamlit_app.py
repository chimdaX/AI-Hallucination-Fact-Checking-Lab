"""Launch with:  streamlit run app/streamlit_app.py   (or: hallab ui)"""

import sys

try:
    import streamlit  # noqa: F401
except ImportError:  # pragma: no cover - only without the optional dependency
    sys.exit('The web interface needs Streamlit. Install it with:  pip install -e ".[ui]"')

try:
    from hallucination_lab.ui.app import main
except ImportError:  # pragma: no cover - package not installed
    sys.exit('Install the project first:  pip install -e ".[ui]"  (from the repository root)')

main()
