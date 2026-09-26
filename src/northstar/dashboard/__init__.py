"""Section 07: executive sales and marketing dashboard built on saved section outputs.

The dashboard never fits models or reads raw tables. It loads the tables and ``metrics.json``
files that the section pipelines write to ``projects/<section>/outputs/``, so every number it
shows can be traced to a file and field (see :mod:`northstar.dashboard.kpis`).
"""

from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")
