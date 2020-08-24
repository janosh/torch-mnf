# Testing Docs

Run all tests by invoking `python -m pytest` from the project's root directory. Requires that `torch_mnf` be installed as a package into the current environment via `pip install .` also from the project's root. See [`setup.py`](../../setup.py).

To run an individual test file, pass a unique identifier as argument, e.g. `python -m pytest **/test_flows.py`.

Note that standard [`pytest` discovery](https://docs.pytest.org/stable/goodpractices.html#conventions-for-python-test-discovery) requires that all functions performing a test be prefixed with `test_`.
