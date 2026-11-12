"""Reviewed Vulture false positives. Static declarations; do not execute.

Do not add internal helpers here to hide unused code. Each exception below is
either consumed by a native library or part of a documented public result.
"""
# Vulture's declaration syntax intentionally uses undefined attribute reads.
# ruff: noqa: F821, B018

# MuJoCo reads these camera and scene-geometry fields from native code.
_.azimuth
_.elevation
_.orthographic
_.emission
_.category

# SiteIKResult: public solver diagnostics for callers (docs/api.md).
_.orientation_error
_.iterations

# RobustCandidateRecord: public per-bank diagnostics (docs/api.md).
_.search_successes
_.validation_successes
_.search_failures
_.validation_failures

# GraspGenerator is a public callable protocol; Python invokes __call__ implicitly.
_.__call__

# Test doubles deliberately fail if a removed viewer HUD API is called.
_.set_texts
_.set_figures
