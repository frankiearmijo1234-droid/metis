# Contributing to metis

Thanks for your interest. metis is a small project but it tries to keep
a high bar on tests, security, and honest documentation. Here's what
contributing looks like.

## Getting set up

```bash
git clone https://github.com/frankiearmijo1234-droid/metis.git
cd metis
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
pytest                                  # should be 200/200
python -m benchmarks.system_test        # should be 29/31 with 2 known pain points
```

If something fails out of the box on your machine, open an issue —
that's a real bug.

## How to contribute

The most useful contributions, in rough order of impact:

1. **Bug reports with a reproducible example.** A small, runnable snippet
   that shows the bug beats a long paragraph every time.
2. **A new engine.** Engines are interchangeable behind the `Engine`
   protocol (see `metis/types.py`). Anything that implements `can_handle`,
   `estimate_cost`, and `solve` is registrable. See "Adding a new engine"
   below.
3. **A new benchmark.** If you have a problem class metis handles poorly,
   adding it to `benchmarks/` helps everyone see where the system breaks.
4. **Better cost estimates.** Most engine cost models are rough heuristics
   calibrated on a single machine. Hardware-specific recalibration is real
   work and very welcome.
5. **Convergence diagnostics for MCMC.** This is a known pain point — see
   the system test phase 5 output. R-hat and effective sample size on
   multi-chain MCMC runs would close a real gap.
6. **Documentation improvements**, especially explaining when an engine
   is the wrong choice. metis is honest about limitations; doc PRs that
   sharpen that honesty are welcome.

## Adding a new engine

A new engine is one Python file in `metis/engines/`. The protocol is
small:

```python
class MyEngine:
    name = "my_engine"

    def can_handle(self, problem: Problem) -> bool:
        """Cheap, side-effect-free check. Return False if you can't
        handle this problem shape; True if you can."""

    def estimate_cost(self, problem: Problem) -> float:
        """Estimated wall-clock seconds. Returning inf is equivalent to
        can_handle returning False."""

    def solve(self, problem: Problem) -> Solution:
        """Actually solve. Validate inputs, enforce caps, return a
        Solution with `value`, `engine_name`, and `elapsed_sec`."""
```

Then register it in `default_router()` in `metis/__init__.py`.

Engine quality checklist before merge:
- Tests in `tests/test_my_engine.py` covering correctness, routing
  eligibility, and validation (bad inputs raise; resource caps enforced).
- Cross-validation against an existing engine where the problem domain
  overlaps (e.g. small QUBO results compared to brute force).
- A line in the README engine table explaining when it wins.
- An entry in `CHANGELOG.md` under `[Unreleased] -> Added`.

## Testing

```bash
pytest                                    # all unit tests
pytest tests/test_my_engine.py            # one file
pytest -v -k "my_test_name"               # one test by name
python -m benchmarks.system_test          # end-to-end
python -m benchmarks.run --quick          # cross-engine perf
```

We aim for tests to run fast (~15s for the full suite) so people actually
run them. If your test takes more than a second, see if it can be made
smaller without losing coverage.

## Style

- **Black formatting**, 88-character lines.
- **Type hints on public APIs**, especially `Engine` methods.
- **Honest comments.** Comments that explain *why* something is the way
  it is are gold. Comments that restate the code are noise.
- **Docstrings on engines** should explain what problems the engine
  handles, what it doesn't handle, and where it wins.

## Security

If you find a security issue, **don't open a public issue**. See
[SECURITY.md](SECURITY.md) for how to report.

Engines and the MCP server have a defense-in-depth validation policy.
Every input crosses at least two layers (MCP server + engine). Resource
caps prevent DoS. No `eval`, `exec`, `pickle`, or `subprocess` in
production code. Contributions that violate this policy will be asked to
change before merge.

## Pull request flow

1. Fork, branch from `main`.
2. Make the change. Run the full test suite. Run the system test.
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. Open a PR with a description that includes:
   - What problem it solves
   - How you tested it
   - Any caveats or known issues

We try to review within a week. Small PRs get merged faster than big ones.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be
kind, be specific, assume good faith.
