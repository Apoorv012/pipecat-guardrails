# Contributing

Thanks for helping. The project is early, so open an issue to discuss larger changes before writing
code.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone the repo, then run `uv sync` and `uv run pre-commit install`.

## Before opening a pull request

- `uv run ruff check . && uv run ruff format --check .`
- `uv run pyright`
- `uv run pytest`
- Add a line under `[Unreleased]` in `CHANGELOG.md` if users will notice the change.
- New public classes and functions need type hints and a docstring (Google style).

## Releases (maintainers)

1. Move the `[Unreleased]` notes in `CHANGELOG.md` under a new version heading.
2. Set the same version in `pyproject.toml`.
3. Commit, then tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
4. The release workflow builds the package, publishes it to PyPI and creates a GitHub release.
