![Logo](/assets/logo.png)

# CytoOne
> A unified probabilistic framework for CyTOF data

![Model Overview](/assets/model_overview.png)

## Installation 

You can easily install `CytoOne` from `PyPI`. Follow the following command to get started:

### Create a virtual environment :snake:

So far, we have only tested the software on Python 3.9 and 3.10.

```shell
conda create -n cytoone python=3.9
conda activate cytoone 
```

or 

```shell
conda create -n cytoone python=3.10
conda activate cytoone 
```

### Build the package

#### pip 

For a stable version of CytoOne, you can download and install the package via 

```shell
pip install CytoOne
```

#### Local 

The latest version of CytoOne will be hosted on GitHub where we constantly update features of the package. To use the latest version of CytoOne, you will need to build the package loacally.

First, you need to clone the repo to a local directory, say `./awesome_repos` and `cd` to that folder. 

Now, you should have a `CytoOne` folder under the `awesome_repos` directory. Run the following to build the package.

```shell 
cd ./CytoOne
python setup.py sdist bdist_wheel
```

You should see a `dist` folder now which contains the wheel file you will need for installing the package. 

```shell
cd ./dist
pip install ./CytoOne-0.0.1-py3-none-any.whl
```


### Dependencies 

With both package building strategies, the dependencies should be installed automatically. Here, we just list them out for your reference.  

- python>=3.9,<3.11
- numpy<2.0
- pandas>=2.2.0
- anndata>=0.10,<0.11
- torch<2.0
- pyro-ppl<1.8.5
- seaborn
- jupyter
- ipywidgets

## Tutorial :fast_forward:

### Interactive Python 
This assumes that you are using Jupyter notebook to run CytoOne.



### Command-Line Interface (CLI)




## Citation :page_with_curl:

If you use CytoOne in your workflow, citing [our paper](https://google.com) is appreciated:

```
@article{
}
```


