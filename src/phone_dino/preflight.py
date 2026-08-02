from __future__ import annotations

import json

from .config import Settings
from .production import ProductionAnalyzer


def main() -> None:
    settings = Settings.from_env()
    if settings.fixture_enabled:
        raise SystemExit("Production preflight rejected: engineering fixtures are enabled")
    ready, reason = ProductionAnalyzer(settings).readiness()
    print(json.dumps({"ready": ready, "reason": reason}, separators=(",", ":"), sort_keys=True))
    if not ready:
        raise SystemExit(1)
