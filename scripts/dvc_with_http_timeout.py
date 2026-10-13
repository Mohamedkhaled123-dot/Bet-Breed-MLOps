"""Run DVC with a bounded default PyDrive2 HTTP timeout."""

import inspect
import sys
from functools import wraps

_TIMEOUT_SECONDS = 60
_EXPECTED_PARAMETERS = (
    ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("settings_file", inspect.Parameter.POSITIONAL_OR_KEYWORD, "settings.yaml"),
    ("http_timeout", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ("settings", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
)


def _install_timeout_override():
    try:
        from pydrive2.auth import GoogleAuth

        original_init = GoogleAuth.__init__
        parameters = tuple(inspect.signature(original_init).parameters.values())
        actual_signature = tuple(
            (parameter.name, parameter.kind, parameter.default)
            for parameter in parameters
        )
        if actual_signature != _EXPECTED_PARAMETERS:
            raise ValueError
    except (ImportError, TypeError, ValueError):
        print(
            "Installed PyDrive2 GoogleAuth signature is incompatible; stopping without running DVC.",
            file=sys.stderr,
        )
        return False

    @wraps(original_init)
    def init_with_timeout(
        self, settings_file="settings.yaml", http_timeout=None, settings=None
    ):
        if http_timeout is None:
            http_timeout = _TIMEOUT_SECONDS
        return original_init(
            self,
            settings_file=settings_file,
            http_timeout=http_timeout,
            settings=settings,
        )

    GoogleAuth.__init__ = init_with_timeout
    return True


def main():
    if not _install_timeout_override():
        return 2

    from dvc.cli import main as dvc_main

    return dvc_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
