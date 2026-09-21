"""Pytest configuration.

Ensures the `backend/` directory is on sys.path so tests can import `app.*`
(e.g. `app.config`, `app.ml.versions`) without requiring an installed package.
"""

import os
import sys

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)