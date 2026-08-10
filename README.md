# weave

## Installation

Clone the repository and enter the project directory:

```bash
git clone git@github.com:Chisen-Lupus/weave.git
cd weave
```

Install the package with:

```bash
python -m pip install .
```

After installation, `weave` can be imported from anywhere in the same Python environment:

```python
import weave
```

### Development installation

If you want to modify the source code locally, install the package in editable mode:

```bash
python -m pip install -e .
```

With an editable installation, changes made to the Python files under `weave/` are reflected automatically without reinstalling the package.

For Jupyter notebooks, it is recommended to enable automatic module reloading during development:

```python
%load_ext autoreload
%autoreload 2

import weave
```

Make sure that `pip` and the Python interpreter refer to the same environment. You can verify the installed package location with:

```bash
python -c "import weave; print(weave.__file__)"
```
