# Publishing flashquad to PyPI

## Prerequisites

- A [PyPI](https://pypi.org/account/register/) account
- An API token from [pypi.org/manage/account/#api-tokens](https://pypi.org/manage/account/token/)

## Build

```bash
uv build
```

This produces a `.tar.gz` and `.whl` in `dist/`.

## Publish

```bash
uv publish --token YOUR_API_TOKEN
```

## Test on TestPyPI first (recommended)

Create a separate account and token at [test.pypi.org](https://test.pypi.org/), then:

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token YOUR_TEST_PYPI_TOKEN
```

Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ flashquad
```

## Bumping the version

Update the `version` field in `pyproject.toml` before each release:

```toml
[project]
version = "0.2.0"
```

## Pre-publish checklist

- [ ] Tests pass (`make test`)
- [ ] Linter is clean (`make check`)
- [ ] Version bumped in `pyproject.toml`
- [ ] README and LICENSE are up to date
- [ ] Dependency bounds in `pyproject.toml` are appropriate for end users
