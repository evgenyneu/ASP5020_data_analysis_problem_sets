# Python code set up

Here is how to install specific versions of Python and its libraries for this project.


## Install Python

1. [Install asdf](https://asdf-vm.com/#/core-manage-asdf).

2. Add python plugin to asdf:

```
asdf plugin add python
```

3. Install Python version specified in [.tool-versions](.tool-versions).

```
asdf install
```


## Create Python environment

```
python -m venv .venv
```

This is put Python and its libraries in local `.venv` directory.


## Activate Python environent

```
source .venv/bin/activate
```

Note: better yet, [setup auto-activation](https://stackoverflow.com/a/50830617/297131).


## Install libraries

```
pip install -r requirements.txt
```

## Deactivate Python environment

When finished working in this directory.

```
deactivate
```


