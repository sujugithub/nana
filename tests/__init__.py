"""Deterministic offline test suite for the learned routing system.

    python3 -m unittest discover -s tests -v

Guarantees enforced by tests/util.py: no network access (a socket guard makes
any connection attempt an error), no model downloads, no paid calls — the
suite must pass on a machine with no API key and no internet.
"""
