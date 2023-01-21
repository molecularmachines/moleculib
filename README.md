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
