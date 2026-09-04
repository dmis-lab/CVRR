"""Dispatch the released CVRR analysis commands."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "localize": "scripts.analysis_cvrr.localization",
    "causal": "scripts.analysis_cvrr.causal.reliance",
    "recurrence": "scripts.analysis_cvrr.recurrence.depth",
    "reread": "scripts.analysis_cvrr.recurrence.reread",
    "persistent": "scripts.analysis_cvrr.recurrence.visual_access",
    "components": "scripts.analysis_cvrr.recurrence.components",
    "residual-swap": "scripts.analysis_cvrr.causal.residual_swap",
    "transition-activation": "scripts.analysis_cvrr.diagnostics.transition",
    "query-channel": "scripts.analysis_cvrr.diagnostics.query_channels",
    "visual-map": "scripts.analysis_cvrr.diagnostics.visual_regions",
    "efficiency": "scripts.analysis_cvrr.efficiency",
    "merge": "scripts.analysis_cvrr.core.merge",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        choices = "\n".join(f"  {name:12s} {module}" for name, module in COMMANDS.items())
        print(
            "Usage: scripts/analyze.sh <command> [args]\n\nCommands:\n"
            + choices
        )
        return 0
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit(
            f"unknown command {command!r}; choose from {', '.join(COMMANDS)}"
        )
    module = importlib.import_module(COMMANDS[command])
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
