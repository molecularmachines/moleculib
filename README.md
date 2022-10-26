# MolecuLib – Molecular Data Pipelining Tools

## Usage

One needs to install moleculib locally with 
```
pip install -e .
```
Then `moleculib` becomes importable in python scripts:

```
from moleculib.protein.batch import PadBatch
```


## Environment
Install [micromamba](https://mamba.readthedocs.io/en/latest/installation.html):

```sh
curl micro.mamba.pm/install.sh | bash #linux
curl micro.mamba.pm/install.sh | zsh  #mac
```

To install listed dependencies and activate the environment:

```sh
micromamba create -f env.yml
micromamba activate moleculib
```

To install **new** dependencies for development:

```sh
micromamba install package_name -c conda-forge
```

To update `env.yml`:
```sh
micromamba env export -n moleculib > env.yml
```


## Testing

Add test as submodule of the module you are developing:

```sh
- module/
    - __init__.py
    - code.py
    - test/
        - __init__.py
        - code_test.py
```

To run all tests:

```sh
python -m unittest discover -s . -p '*_test.py'
```

To run tests from a specific module:

```sh
python -m unittest discover -s modulepath -p '*_test.py'
```


## Linting

```sh
black .
```
