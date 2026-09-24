# Contributing to PhysAI

Thank you for considering a contribution. This guide describes how to propose changes that are clear, reproducible, and straightforward to review.

## Before you start

- Search the issue tracker and existing pull requests to see whether the work is already being discussed.
- For a substantial change, open an issue first to agree on the intended behavior and scope.
- For a security vulnerability, follow [SECURITY.md](SECURITY.md) and report it privately instead of opening a public issue.
- Keep changes focused. Explain any numerical, API, or compatibility tradeoffs in the issue or pull request.

## Development setup

PhysAI supports Python 3.9 and newer. From a local clone, install the package in editable mode and pytest:

```bash
python -m pip install -e .
python -m pip install pytest
```

Optional framework backends and features are installed through extras. Install only the extra needed for the area you are working on; some backends and native solver integrations have substantial or platform-specific dependencies. See the installation section in [README.md](README.md) and the optional dependency groups in `pyproject.toml`.

## Tests and validation

Add or update tests for behavior changes, bug fixes, and new public APIs. Prefer focused tests that cover both expected behavior and relevant invalid inputs. For example:

```bash
python -m pytest tests/test_geometry.py
```

Run the relevant focused test file while iterating, then run the full suite before submitting when the installed dependencies permit it:

```bash
python -m pytest
```

If a test requires an optional backend or external solver that is unavailable in your environment, say so in the pull request and include the validation you were able to perform. Do not describe unrun tests as passing.

For changes to PDE residuals or numerical methods, include the equation, assumptions, coordinate and field conventions, boundary or initial conditions, and a reference where applicable. Check shapes, dtypes, device behavior, and finite outputs across the relevant backend(s). For user-facing examples, make sure the documented calls match the public API.

## Pull requests

1. Create a branch for the change.
2. Make the smallest coherent change and update relevant documentation or examples.
3. Run the applicable checks and record the exact commands and results.
4. Open a pull request against the project’s default branch. Describe the problem, the chosen solution, compatibility implications, and any limitations. Link related issues and include screenshots or numerical comparisons when they clarify the change.
5. Keep the pull request focused and respond constructively to review feedback.

Maintainers may ask for revisions, tests, or a narrower scope before merging. Please do not add generated files, local environment data, credentials, or unrelated formatting changes.

## Style and compatibility

- Follow the existing Python style and naming conventions in the affected module.
- Preserve existing public behavior unless the change intentionally updates it; document intentional API changes.
- Validate inputs at public boundaries and use actionable error messages.
- Keep optional dependencies optional. Do not make importing the base package require an optional backend or external solver.
- Keep numerical changes reproducible and explain solver settings or tolerances that materially affect results.

## Licensing and conduct

PhysAI is distributed under the GNU Affero General Public License, version 3 only (AGPL-3.0-only). By submitting a contribution, you agree that it may be distributed under that license. Do not submit code or other material unless you have the right to contribute it.

All contributors are expected to follow the project’s [Code of Conduct](CODE_OF_CONDUCT.md).
