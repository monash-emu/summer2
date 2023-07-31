# summer2: compartmental disease modelling in Python

[![Automated Tests](https://github.com/monash-emu/summer2/actions/workflows/tests.yml/badge.svg)](https://github.com/monash-emu/summer2/actions/workflows/tests.yml)

summer2 is a Python-based framework for the creation and execution of [compartmental](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) (or "state-based") epidemiological models of infectious disease transmission.

It provides a range of structures for easily implementing compartmental models, including structure for some of the most common features added to basic compartmental frameworks, including:

- A variety of inter-compartmental flows (infections, transitions, births, deaths, imports)
- Force of infection multipliers (frequency, density)
- Post-processing of compartment sizes into derived outputs
- Stratification of compartments, including:
  - Adjustments to flow rates based on strata
  - Adjustments to infectiousness based on strata
  - Heterogeneous mixing between strata
  - Multiple disease strains

Some helpful links to learn more:

- **[Documentation](https://summer2.readthedocs.io/)** with [code examples](https://summer2.readthedocs.io/en/latest/examples/index.html)
- [Available on PyPi](https://pypi.org/project/summerepi2/) as `summerepi2`.

## Installation and Quickstart

This project requires at least Python 3.8

Set up and activate an appropriate virtual environment, then install the `summerepi2` package from PyPI

```bash
pip install summerepi2
```

Important note for Windows users:
summerepi2 relies on the Jax framework for fast retargetable computing.  This is automatically
installed under Linux, OSX, and WSL environments. 
It is strongly recommended that you use WSL, but in instances were you are unable to do so,
an unofficial build of jax can be installed by running the following command

```bash
pip install jax[cpu]==0.3.24 -f https://whls.blob.core.windows.net/unstable/index.html
```

Then you can now use the library to build and run models. See [here](https://summer2.readthedocs.io/en/latest/examples/index.html) for some code examples.

## Optional (recommended) extras

Summer has advanced interactive plotting tools built in - but they are greatly improved with the
addition of the pygraphviz library.

If you are using conda, the simplest method of installation is as follows:

```bash
conda install --channel conda-forge pygraphviz
```

For other install methods, see
https://pygraphviz.github.io/documentation/stable/install.html

## Development

[Poetry](https://python-poetry.org/) is used for packaging and dependency management.

Initial project setup is documented [here](./docs/dev-setup.md) and should work for Windows or Ubuntu, maybe for MacOS.

Some common things to do as a developer working on this codebase:

```bash
# Activate summer conda environment prior to doing other stuff (see setup docs)
conda activate summer

# Install latest requirements
poetry install

# Publish to PyPI - use your PyPI credentials
poetry publish --build

# Add a new package
poetry add

# Run tests
pytest -vv

# Format Python code
black .
isort . --profile black
```

## Releases

Releases are numbered using [Semantic Versioning](https://semver.org/)

- 1.0.0/1:
  - Initial release
- 1.2.1
  - Dropped support for Python 3.7.  Variety of bugfixes and expanded features, see documentation
- 1.2.2
  - Added capture utils
  - Removed inner jit in model building to improve debugging
- 1.2.4
  - CompartmentValues not functioning previously; include more comprehensive check in graph split
- 1.2.5
  - Previous issue was not resolved correctly; bugfix-bugfix release
- 1.2.6
  - Make full (comparment) outputs optional for jax runners
- 1.2.8
  - Improve time varying functions (allow GraphObjects in x values)
- 1.2.9
  - Derived output functions matching pandas.Series functionality (rolling, diff)

## Release process

To do a release:

- Commit any code changes and push them to GitHub
- Choose a new release number accoridng to [Semantic Versioning](https://semver.org/)
- Add a release note above
- Edit the `version` key in `pyproject.toml` to reflect the release number
- Publish the package to [PyPI](https://pypi.org/project/summerepi/) using Poetry, you will need a PyPI login and access to the project
- Commit the release changes and push them to GitHub (Use a commit message like "Release 1.1.0")
- Update `requirements.txt` in Autumn to use the new version of Summer

```bash
poetry build
poetry publish
```
