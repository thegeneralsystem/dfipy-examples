# The Data Flow Index Platform API Guide

[![Python versions](https://img.shields.io/pypi/pyversions/dfipy.svg)](https://pypi.python.org/pypi/dfipy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)

<img src="docs/gs_logo.png" width="40%" align="right">

This document will guide you through the steps necessary to try out the Platform
API from [General System](https://www.generalsystem.com) using the GS client-side Python library, [dfipy](https://github.com/thegeneralsystem/dfipy). The library provides a layer of abstraction over the [DFI Web API](https://api.dataflowindex.io/docs/api), including presenting data in the popular [Pandas](https://pandas.pydata.org/) dataframe format.

- [`dfipy`](https://github.com/thegeneralsystem/dfipy) documentation is available at <https://dfipy.docs.generalsystem.com/index.html>.

- Additional resources and help are available at <https://support.generalsystem.com>.

## Overview

This repository contains example Python snippets in the form of
[Jupyter Notebooks](https://jupyter.org/) that demonstrate connecting to and
querying of public demonstration servers.

## Step 1: Obtain your API token

Contact General System to obtain an API token.

## Step 2: Launch Jupyter Notebook

The Jupyter Notebooks may be downloaded and installed
locally, or alternatively there are a number of free online options.

### I. Running Online

Open the Quick Start guides directly in [Google Colab](https://colab.research.google.com/) or [Binder](https://mybinder.org/):

**Colab:**

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thegeneralsystem/dfipy-examples/blob/main/examples/dfipy_quick_start_syn_traffic.ipynb) - Query a large synthetic traffic dataset of 92 billion records using a streaming API.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thegeneralsystem/dfipy-examples/blob/main/examples/dfipy_quick_start_create_a_polygon.ipynb) - A beginner's guide to draw polygons on a map and then query the DFI.

**Binder:**

- [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/thegeneralsystem/dfipy-examples/HEAD?labpath=examples%2Fdfipy_quick_start_syn_traffic.ipynb) - Query a large synthetic traffic dataset of 92 billion records using a streaming API.
- [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/thegeneralsystem/dfipy-examples/HEAD?labpath=examples%2Fdfipy_quick_start_create_a_polygon.ipynb) - A beginner's guide to draw polygons on a map and then query the DFI.

### II. Running locally

Follow the installation instructions at <https://jupyter.org/install#jupyter-notebook>:

This repository is based on the python library `dfipy` , publicly installable with `pip install dfipy`.

Within the target virtual environment (which can be a virtualenv, a conda env, or a poetry env according to your preferences), install the requirements with:

```bash
pip install -r requirements.txt
jupyter notebook
```

### III. Running Locally from Dockerfile

```bash
docker build -t binder-image .
docker run -it --rm -p 8888:8888 binder-image jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888
```

## Step 3: Running the Example Notebooks

Once you have opened the example notebooks simply follow the
guidance contained within each notebook. You will need to provide your Platform API
token to run the code.

The example notebooks in the `examples/` folder are as follows:

| Notebook                             | Description                                                                                                                                                                           |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dfipy_quick_start_syn_traffic`      | Query a large synthetic traffic dataset of 92 billion records using a streaming API                                                                                                   |
| `dfipy_quick_start_create_a_polygon` | A beginner's guide to draw polygons on a map and then query the DFI                                                                                                                   |

## Licence

- Copyright (c) 2023, General System Group Limited.

- DFIPyExamples is provided as it is and copyrighted under [Apache2 License](LICENCE).

- DFIPyExamples is publicly available on github strictly for testing and evaluation purposes.
