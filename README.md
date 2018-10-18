# Develer turns to Data Science

## Lecture notes for the "Data Science, the Pythonic way" @ [Develer](https://www.develer.com/)

<img src="http://bit.ly/develer_logo" title="Develer Logo" width="20%" />

### Author: Valerio Maggio

#### _PostDoc Data Scientist @ FBK/MPBA_

#### Contacts:

<table style="border: 0px; display: inline-table">
    <tbody>
        <tr style="border: 0px;">
            <td style="border: 0px;">
                <img src="images/twitter_small.png" style="display: inline-block;" />
                <a href="http://twitter.com/leriomaggio" target="\_blank">@leriomaggio</a>
            </td>
            <td style="border: 0px;">
                <img src="images/linkedin_small.png" style="display: inline-block;" />
                <a href="it.linkedin.com/in/valeriomaggio" target="\_blank">valeriomaggio</a>
            </td>
            <td style="border: 0px;">
                <img src="images/gmail_small.png" style="display: inline-block;" />
                valeriomaggio_at_gmail_dot_com
            </td>
       </tr>
  </tbody>
</table>

# Materials:

![github](./images/github.jpg)

```shell
git clone https://github.com/leriomaggio/develer-data-science.git
```

# Outline at a glance:
(from _apprentice_ to _doctor strange_)

- **Level I**) _Apprentice_:  **Pythonic tools for Data Science**

    * _Dev Tools_ for Data Scientist and Jupyter notebooks
    * Numerical computation in Python: `numpy`
    * Working with data: `pandas`


- **Level II**) _Alchemist_: **Data Visualisation**

    * Basic principles of data visualisation
    * Introduction to `matplotlib`
    * interactive data visualisation using `bokeh`


- **Level III**) _Mage_: **Crash course on Machine Learning**

    * What is _Machine Learning_
    * Introduction to `sklearn`
    * _Supervised_ and _**Un**supervised_ Machine learning
    * Robust Machine Learning: _selection bias and cross-validation_


- **Level IV**) _Arch-Mage_ : **Deep Learning & Pythonic perspectives**
    * What is _Deep Learning_
    * Deep Learning frameworks
    * Introduction to Keras

### Description

The course will be organised in **four** different parts,
mostly covering the basics (plus some more advanced topics)
related to Machine Learning and Data Science.

We will start by introducing the basics of data science in Python,
and the (development) tools and frameworks to be used.
Then we will start working with real data (in different formats)
to have a very general feeling of what does it _mean_ to be
a _data scientist_. There will also be a section specifically
focused on basic principles (and tools) of
data visualisation.
Finally, more advanced concepts will be introduced.
In particular, a general introduction to Machine Learning models
and settings (i.e. _supervised_ and _unsupervised_) will be
provided, along with a glimpse of Deep learning models and
frameworks.

All these parts will be presented always considering the
perspective of the developer and practitioner who wants to
learn (and understand) _Data Science_ in a very practical way.
For this aim, the materials will contain lots of
exercises and challenges along the way to test your
skills.

---

# Technical Requirements

This tutorial requires the following packages:

- Python version 3.6
    - Python 3.4+ should be fine as well
    - likely Python 2.7 would be also fine, but *who knows*? :P
- `numpy`: http://www.numpy.org/
- `scipy`: http://www.scipy.org/
- `matplotlib`: http://matplotlib.org/
- `pandas`: http://pandas.pydata.org
- `scikit-learn` : http://scikit-learn.org
- `jupyter` & `notebook`: http://jupyter.org

Plus - for the last Deep learning section:
- `keras`: http://keras.io
- `tensorflow`: https://www.tensorflow.org
- (optional) `torch`: http://pytorch.org

The easiest way to get (most of) these is to use an all-in-one installer
such as [Anaconda](https://www.anaconda.com/download/) from Continuum,
which is available for multiple computer platforms, namely Linux,
Windows, and OSX.

---

### Python Version

I'm currently running this tutorial with **Python 3** on **Anaconda**


```shell
$ python --version
Python 3.6.6
```

---

# Accessing the materials

If you want to access the materials, you have several options:

## Jupyter Notebook

Most of the materials in this course is provided as a collection of
Jupyter Notebooks.

In case you don't know **what is** a Jupyter notebook, here is a good
reference for a quick introduction:
[Jupyter Notebook Beginner Guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).

On the other hand, if you also want to know (_and you should_) **what is NOT**
a Jupyter notebook - *spoiler alert:* **it is NOT an IDE** -
here is a very nice reference:

&rightarrow; [I Don't like Notebooks,](https://twitter.com/joelgrus/status/1033035196428378113)
by _Joel Grus_ @ JupyterCon 2018.

If you **already have all the environment setup** on your machine,
all you need to do is to run the Jupyter notebook server:

```shell
$ jupyter notebook
```

Alternatively, I suggest you to try the new **Jupyter Lab** environment:
```shell
$ jupyter lab
```

**NOTE**: Before running Jupyter server, it is mandatory to enable
the (Python) virtual environment.

Please refer to the section [Setting the Environment](#setup) for
detailed instructions on how to install all the required
packages and libraries.


## Binder

(Consider this option only if your WiFi is stable)

If you don't want the hassle of setting up all the environment and
libraries on your machine, or simply you want to avoid doing
"_too much computation_" on your hardware setup,
I strongly suggest you to use the **Binder** service.

The primary goal of Binder is to turn a GitHub repo into a collection of
interactive Jupyter notebooks

To start using Binder, just click on the button below:
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/leriomaggio/develer-data-science/master)

## Google Colaboratory

[Colaboratory](https://colab.research.google.com/) is a free Jupyter
notebook environment that
requires no setup and runs entirely in the Google cloud.
Moreover, **GPU** and **TPU** runtime environments are available,
and completely for free.
(This last option will be worthwhile mentioning in the very
last part of the course, when we will talk
about Deep Learning networks).

[Here](https://colab.research.google.com/notebooks/welcome.ipynb)
is an overview of the main features offered by Colaboratory.

To start using Colaboratory, just click on the button below:
[![Colab](https://img.shields.io/badge/launch-colaboratory-yellow.svg)](https://colab.research.google.com/)

---

<a name='setup'></a>
# Setting the Environment

In this repository, files to install the required packages are provided.
The first step to setup the environment is to create a
Python [Virtual Environment](https://docs.python.org/3.6/tutorial/venv.html).

Whether you are using [Anaconda](https://www.anaconda.com/download/)
Python Distribution or the **Standard
Python framework** (from [python.org](https://www.python.org/downloads/)),
below are reported the instructions for the two cases, respectively.

## (a) Conda Environment

This repository includes a `conda-environment.yml` file that is necessary
to re-create the Conda virtual environment.

To re-create the virtual environments:

```shell
$ conda env create -f conda-environment.yml
```

Then, to **activate** the virtual environment:

```shell
$ conda activate develer-science
```

## (b) `pyenv` & `virtualenv`

Alternatively, if you don't want to install (yet) another Python
distribution on your machine, or you prefer not to use the full-stack Anaconda
Python, I strongly suggest to give a try to the new `pyenv` project.

### 1. Setup `pyenv`

`pyenv` is a new package that lets you easily switch between multiple
versions of Python.
It is simple, unobtrusive, and follows the UNIX tradition of single-purpose
tools that do one thing well.

To **setup** `pyenv`, please follow the instructions reported on the
[GitHub Repository](https://github.com/pyenv/pyenv) of the project,
according to the specific platform and operating system.

There exists a `pyenv` plugin named `pyenv-virtualenv` which comes with various
features to help `pyenv` users to manage virtual environments created by
`virtualenv` or Anaconda.

### 2. Installing `pyenv-virtualenv`

I would recommend to install `pyenv-virtualenv` as reported in
the official
[documentation](https://github.com/pyenv/pyenv-virtualenv/blob/master/README.md).

### 3. Setting up the virtual environment

Once `pyenv` and `pyenv-virtualenv` have been correctly installed and
configured, these are the instructions to
set up the virtual environment for this tutorial:

```shell
$ pyenv install 3.6.6  # downloads and enables Python 3.6
$ pyenv virtualenv 3.6.6 develer-science  # create virtual env using Py3.6
$ pyenv activate develer-science  # activate the environment
$ pip install -r requirements.txt  # install requirements

```

### Installing Jupyter Kernel (Optional)

All the notebooks in this tutorial have been saved using a Jupyter Kernel
defined on the created virtual environment, named "Python 3.6 (DL Keras TF)".

In case you got a warning of _non-existent kernel_ when you open the
notebooks on your machine, you need to create the corresponding
`IPython` kernel:

```shell
$ python -m ipykernel install --user --name develer-science --display-name "Python 3.6 (Develer Science)"
```

---

## Test if everything is up&running

### 1. Check import


```Python
>>> import numpy as np
>>> import scipy as sp
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import sklearn
>>> import keras
Using TensorFlow backend.
```

### 2. Check installed Versions


```Python
>>> import numpy
>>> print('numpy:', numpy.__version__)
>>> import scipy
>>> print('scipy:', scipy.__version__)
>>> import matplotlib
>>> print('matplotlib:', matplotlib.__version__)
>>> import sklearn
>>> print('scikit-learn:', sklearn.__version__)
```
```
    numpy: 1.15.2
    scipy: 1.1.0
    matplotlib: 3.0.0
    scikit-learn: 0.20.0
```

<br>
<h2 style="text-align: center;">If everything worked till down here, you're ready to start!</h2>
