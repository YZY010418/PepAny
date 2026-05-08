#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Lightweight CLI wrapper for PepAny generation."""

try:
    from .generation_cli import parse
except ImportError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from api.generation_cli import parse


def _runtime():
    try:
        from . import generate_runtime
    except ImportError:
        import importlib
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        generate_runtime = importlib.import_module('api.generate_runtime')
    return generate_runtime


def main(args, opt_args):
    return _runtime().main(args, opt_args)


def __getattr__(name):
    return getattr(_runtime(), name)


if __name__ == '__main__':
    args, opt_args = parse()
    _runtime().print_log(f'Overwritting args: {opt_args}')
    main(args, opt_args)
