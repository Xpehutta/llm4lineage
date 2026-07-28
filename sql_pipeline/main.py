"""CLI shim — use ``python -m Classes.pipeline.main`` or ``sql-pipeline``."""

from Classes.pipeline.main import main

if __name__ == "__main__":
    raise SystemExit(main())
