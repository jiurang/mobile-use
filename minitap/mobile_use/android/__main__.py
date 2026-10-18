"""Entry point: ``python -m minitap.mobile_use.android``."""

from __future__ import annotations

import argparse

from minitap.mobile_use.android.server import main as serve


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="mobile-use-android",
        description="Run the mobile-use agent on this Android device and serve its UI over HTTP.",
    )
    parser.parse_args()
    serve()


if __name__ == "__main__":
    cli()
