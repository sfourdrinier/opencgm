"""Mark `scripts/` as a Python package so the parity tests can reuse the export wrappers.

Each script here is also runnable as `python scripts/<name>.py`, but tests want to import
their helpers (`EncoderMeanEmbed`, the decomposed transformer) as ordinary modules. Adding
this empty marker is the cheapest way to support both.
"""
