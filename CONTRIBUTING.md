# Contributing

Start with the [development guide](docs/development.md) for architecture,
extension examples, frame invariants and validation commands.

```bash
pip install -e ".[dev,docs]"
pytest
ruff check src tests examples tools
ruff format --check src
vulture
```

Keep public imports compatible, preserve the fixed-world-command invariant,
and add regressions for changed behavior. Simulator changes also need their
optional dependencies and a physical demo run. Describe the caller-visible
change and the validation actually performed in each pull request.
