"""Experiment tooling.

``paths`` is the single place that knows the record/artifact layout; ``report``
turns a directory of completed runs into ``summary.csv``, ``REPORT.md`` and
``comparison.png``. Nothing here is an entry point: the CLI lives in
``manage.py``.
"""
