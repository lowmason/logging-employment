"""§11's Bayesian models. Stage 5 ships the state-total model; Stage 6 adds the size model.

NOTHING HERE IMPORTS JAX AT PACKAGE IMPORT. `interfaces`, `data`, `reconciliation` and `summary`
are plain NumPy and polars, so the CLI, the harness and the promotion gate can import them without
initialising a JAX backend (§16.2: "PPL-specific objects must remain behind model interfaces").
`state_total` is the only module that imports NumPyro and JAX, and `arviz_io` the only one that
imports ArviZ and xarray. `diagnostics` reaches ArviZ through `arviz_io`, and `validation` reaches
both through the fit, so `cli.py` imports each of those inside the command that needs it.
"""
