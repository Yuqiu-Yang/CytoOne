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

Remember to change the `VERSION` to match the file you see in the `dist` folder.

```shell
cd ./dist
pip install ./CytoOne-VERSION-py3-none-any.whl
```

Once you have installed the package, you can quickly test if the installation is successful via 

```shell
python -m CytoOne --version
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

## Input format 

The input for CytoOne consists of two pieces of data: 

1. `cell_by_gene`: An optional dataframe of the cell-by-gene matrix. 
    * The column names should be genes/protein markers
    * The first column is assumed to contain IDs of cells 
    * Note that `CytoOne` is designed to model the asinh-transformed protein measurements. Based on our experience, the range is usually around 0 to 10. If you spot values in the hundreds, toggle `normalize` option on. Otherwise, set `normalize=False`.
2. (Optional) `cell_metadata`: An optional dataframe containing meta information on the cells. 
    1. `cell_id`: IDs of segmented cells. MisTIC assumes that the first column of the dataframe will be the cell IDs. Make sure that the cell IDs correspond to those recorded in the `cell_by_gene` dataframe.
    2. `batch`: Batch annotations for the cells. 
    3. (Optional) `cell_type`: Annotated cell types. This is not used in the algorithm, but could be helpful when plotting the results. 
    * If `cell_metadata` is not provided, `CytoOne` would internally generate one by assuming all cells were generated in one single batch. 
    * Note that the column names in your dataset might not be exactly what we specified here, but you can inform `CytoOne` the names of your dataset that can be used as those four columns. 

We have also included in this package a sample dataset under `tests` for your reference. 

## Tutorial :fast_forward:

### Interactive Python 
This assumes that you are using Jupyter notebook to run CytoOne.

1. Object instantiation

```python
>>> from CytoOne.cytoone_class import cytoone
>>> # Check and specify the column names!
>>> cyto = cytoone(batch_index_col='BATCH COLUMN IN CELLMETA',
                   celltype_col='CELLTYPE COLUMN IN CELLMETA',
                   normalize='CHECK YOUR CELL-BY-GENE',
                   zero_inflated='CHECK YOUR CELL-BY-GENE',
                   dr=True)
```

* You should check your `cell-by-gene` matrix to make sure if the measurement has been normalized and is zero-inflated.
* If `normalize=True`, `CytoOne` will transform the data via asinh with a cofactor of 5.
* if `zero_inflated=True`, `CytoOne` will truncate the data below 0 to make sure no negative values are present. 

Set `dr=False` if you do not need UMAP. 

2. Data importing 

Once the object is instantiated, we are now ready to load the data into the object. 

```python
>>> cell_by_gene = "PATH/TO/CELL-BY-GENE"
>>> cell_metadata = 'PATH/TO/META'
>>> cyto.import_data(cell_by_gene=cell_by_gene,
                    cell_metadata=cell_metadata)
```

This will not only import the data into the `cyto` object, but also perform some data curation. 

* Note that you can read in the dataframes first, and then simply use them as input to the function instead of paths to the files. This however, will unnecessarily incease memory usuage as CytoOne will first create deep copies of the supplied dataframes. 

3. Model training

```python
>>> # Modle training 
>>> cyto.initialize_parameters()
>>> cyto.training_loop()
```

* At this stage, there are other arguments you can use to tweak the training behavior of `CytoOne`. Although the default setting works well based on our experience, you are welcome to change their settings: 
    1. `n_epoches`: The default number of iteration is 50. 
    2. `n_strata`: Number of minibatches per epoch. 
    3. `early_stop_pval`: If `n_epoches>=3`, starting from the third epoch, `CytoOne` will check if the reconstruction loss has stablized via the KS-test. Once the returning p-value is greater than `early_stop_pval` the training will stop. By default, this is disabled by setting `early_stop_pval=1`.

4. Downstream analyses

Once the model is trained, there are a few downstream analyses that you can perform with the `infer` function. 

* Dimension reduction

To get the embedding for the current dataset, you can simply call: 

```python
>>> _, z_samples = cyto.infer()
```

If you want to project a new dataset using the trained model, simply supply `cyto` with the new data by: 

```python 
>>> _, z_samples = cyto.infer(new_cell_by_gene="PATH/TO/NEW/CELL-BY-GENE",
                              new_cell_metadata='PATH/TO/NEW/CELL-METADATA')
```

The resulting `z_samples` will be a dataframe with 2 columns which can be used directly for visualization. 

* Batch normalization 

CytoOne allows you to normalize all data to a reference batch. Simply use 

```python 
>>> x_samples, z_samples = cyto.infer(target_batch_index=0)
```

You can set `target_batch_index` to whichever batch you want. `x_samples` will contain normalized protein measurements whereas `z_samples` will be the corresponding latent embeddings. 

* Differential expression analysis 

CytoOne uses the normal components of the QZIPN distribution for DE analysis by using 

```python 
>>> x_samples, _ = cyto.infer(get_normal_component=True)
```

The `x_samples` will contain the samples from the underlying normal distributions of the protein measurements. 


5. Model saving 

To save the model, simply do the following: 

```python
>>> cyto.save_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="cyto")
```

This will save PyTorch model `cyto.pt` along with some meta information `cyto_meta.json`. 

6. Model loading 

Loading the saved model could be useful if you turned off the program but wanted to take a closer look at the results later on. As previously stated, since not all information is saved with the model, you will need to import the data again.

```python
>>> from CytoOne.cytoone_class import cytoone
>>> # Check and specify the column names!
>>> cyto = cytoone("MAKE/SURE/TO/CHECK/COLUMN/NAMES!!!",
                  model_device="cpu")
>>> # Load the model 
>>> cyto.load_model(dir_name="PATH/TO/DIRECTORY",
                    model_name="cyto")
>>> cell_by_gene = "PATH/TO/CELL-BY-GENE"
>>> cell_metadata = 'PATH/TO/META'
>>> cyto.import_data(cell_by_gene=cell_by_gene,
                    cell_metadata=cell_metadata)
>>> x_samples, z_samples = cyto.infer()
```

That's it~


### Command-Line Interface (CLI)

The arguments for CLI is almost identical to those in the interactive Python.


```shell
python -m CytoOne --batch_index_col batch
                --celltype_col cell_type
                --cell_by_gene PATH/TO/CELL-BY-GENE
                --cell_metadata PATH/TO/META
                --dir_name .
                --model_name cyto
```

* For a comprehensive list of arguments, you can use 

```python 
python -m CytoOne -h
```

## Citation :page_with_curl:

If you use CytoOne in your workflow, citing [our paper](https://google.com) is appreciated:

```
@article{
}
```


