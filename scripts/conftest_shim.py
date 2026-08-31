"""Re-export the tests' StubEnv so scripts can drive reward terms off-GPU.

The stub lives with the tests because that is where it is maintained; this
keeps `scripts/` from growing a second, drifting copy of the same fake.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import StubEnv, amplitudes, twist_at  # noqa: E402,F401
