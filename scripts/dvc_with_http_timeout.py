
"""Run DVC with a bounded default PyDrive2 HTTP timeout."""

import sys
from functools import wraps


_TIMEOUT_SECONDS = 60


def _install_timeout_override():
    try:
        from pydrive2.auth import GoogleAuth
    except ImportError as exc:
        print(f"Cannot import PyDrive2 GoogleAuth: {exc}", file=sys.stderr)
        return False

    original_init = GoogleAuth.__init__

    @wraps(original_init)
    def init_with_timeout(self, *args, **kwargs):
        if kwargs.get("http_timeout") is None:
            kwargs["http_timeout"] = _TIMEOUT_SECONDS
        return original_init(self, *args, **kwargs)

    GoogleAuth.__init__ = init_with_timeout
    return True


def main():
    if not _install_timeout_override():
        return 2

    from dvc.cli import main as dvc_main

    return dvc_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
