# Data IO 
import os 
from copy import deepcopy
import numpy as np 
import pandas as pd 

# PyTorch
import torch  
from torch.distributions.kl import register_kl
from pyro.distributions import Delta 
from torch.distributions import Normal, Categorical
from CytoOne.basic_distributions import zero_inflated_lognormal

# Typing 
from typing import Optional, Tuple, Union


################################
# New KL divergences 
# The formula is KL(p||q) = -\int p \log \dfrac{q}{p} 

@register_kl(Delta, Delta)
def _kl_delta_delta(p, q) -> torch.tensor:
    return -q.log_density 

@register_kl(Delta, zero_inflated_lognormal)
def _kl_delta_ziln(p, q) -> torch.tensor:
    return -q.log_prob(p.v) 

@register_kl(Delta, Categorical)
def _kl_delta_categorical(p, q) -> torch.tensor:
    return -q.log_prob(p.v) 


############################
def data_curation(user_data: Union[pd.DataFrame, str], 
                  batch_index_col_name: Optional[str]=None,
                  condition_index_col_name: Optional[str]=None,
                  subject_index_col_name: Optional[str]=None,
                  cell_type_index_col_name: Optional[str]=None,
                  other_nonprotein_col_names: Optional[list]=None,
                  batches_to_retain: Optional[list]=None,
                  conditions_to_retain: Optional[list]=None,
                  subjects_to_retain: Optional[list]=None,
                  cell_types_to_retain: Optional[list]=None,
                  is_zero_inflated: bool=True,
                  arcsinh5_transform: bool=True,
                  **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Curate CyTOF dataset
    
    This function curates a CyTOF dataset and returns the curated data and 
    the meta information needed for model input 

    Parameters
    ----------
    user_data : Union[pd.DataFrame, str]
        Either a pandas dataframe of or the path to the dataset to be used 
    batch_index_col_name : Optional[str], optional
        The name of the column that indexes batches. 
        If None, we assume all data is from one batch, by default None
    condition_index_col_name : Optional[str], optional
        The name of the column that indexes conditions (treatments). 
        If None, we assume all data is from one condition, by default None
    subject_index_col_name : Optional[str], optional
        The name of the column that indexes subjects (patients). 
        If None, we assume all data is from one subject, by default None
    cell_type_index_col_name : Optional[str], optional
        The name of the column that indexes cell types. 
        This is only used for validation/visualization, by default None
    other_nonprotein_col_names : Optional[list], optional
        The names of columns that provide other information that are not going to be considered by the model. 
        If None, all columns that are not batch, condition, subject, or cell type will be deemed as protein channels, by default None
    batches_to_retain : Optional[list], optional
        A list of batch indices to retain. If None, all batches are kept, by default None
    conditions_to_retain : Optional[list], optional
        A list of condition indices to retain. If None, all conditions are kept, by default None
    subjects_to_retain : Optional[list], optional
        A list o subject indices to retain. If None, all subjects are kept, by default None
    cell_types_to_retain : Optional[list], optional
        A list of cell types to retain. If None, all cell types are kept, by default None
    is_zero_inflated : bool, optional
        Whether or not the data is zero inflated. If True, we will truncate the data from below 0, by default True
    arcsinh5_transform : bool, optional
        Whether or not we need to transform the data using the inverse hyperbolic sine function with factor of 5, by default True

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        The first output is the curated dataframe, while the second one contains meta information of the data. 
    """
    
    # We first make sure that the input is legitmate 
    assert isinstance(batch_index_col_name, str) or (batch_index_col_name is None), "batch_index_col_name should be a string or None"
    assert isinstance(condition_index_col_name, str) or (condition_index_col_name is None), "condition_index_col_name should be a string or None"
    assert isinstance(subject_index_col_name, str) or (subject_index_col_name is None), "subject_index_col_name should be a string or None"
    assert isinstance(cell_type_index_col_name, str) or (cell_type_index_col_name is None), "cell_type_index_col_name should be a string or None"
    assert isinstance(other_nonprotein_col_names, list) or (other_nonprotein_col_names is None), "other_nonprotein_col_names should be a list of strings or None"
    # If ~~~to_retain is specified but the corresponding column names are not. 
    assert (batches_to_retain is None) or (batch_index_col_name is not None), "Please specify batch index col"
    assert (conditions_to_retain is None) or (condition_index_col_name is not None), "Please specify condition index col"
    assert (subjects_to_retain is None) or (subject_index_col_name is not None), "Please specify subject index col"
    assert (cell_types_to_retain is None) or (cell_type_index_col_name is not None), "Please specify cell type index col"
    # Read data 
    if isinstance(user_data, str):
        df = pd.read_csv(user_data, **kwargs)
    else:
        df = deepcopy(user_data)
    # We filter the data frame according to ~~to_retain
    if batches_to_retain is not None:
        df = df[df[batch_index_col_name].isin(batches_to_retain)].reset_index(drop=True)
    if conditions_to_retain is not None:
        df = df[df[condition_index_col_name].isin(conditions_to_retain)].reset_index(drop=True)
    if subjects_to_retain is not None:
        df = df[df[subject_index_col_name].isin(subjects_to_retain)].reset_index(drop=True)
    if cell_types_to_retain is not None:
        df = df[df[cell_type_index_col_name].isin(cell_types_to_retain)].reset_index(drop=True)
    # As various factors might not be labels like 0, 1, 2, ....
    # we convert the indices to nature numbers 
    nonprotein_col_names = []
    if batch_index_col_name is not None:
        batch_index, _ = pd.factorize(df[batch_index_col_name])
        nonprotein_col_names.append(batch_index_col_name)
    else:
        batch_index = np.zeros(df.shape[0], dtype=np.int32)
    if condition_index_col_name is not None:
        condition_index, _ = pd.factorize(df[condition_index_col_name])
        nonprotein_col_names.append(condition_index_col_name)
    else:
        condition_index = np.zeros(df.shape[0], dtype=np.int32)
    if subject_index_col_name is not None:
        subject_index, _ = pd.factorize(df[subject_index_col_name])
        nonprotein_col_names.append(subject_index_col_name)
    else:
        subject_index = np.zeros(df.shape[0], dtype=np.int32)
    if cell_type_index_col_name is not None:
        cell_type_index, _ = pd.factorize(df[cell_type_index_col_name])
        nonprotein_col_names.append(cell_type_index_col_name)
    else:
        cell_type_index = np.zeros(df.shape[0], dtype=np.int32)
    
    if other_nonprotein_col_names is not None:
        nonprotein_col_names += other_nonprotein_col_names
    # Everything else is then protein channels
    protein_col_names = list(df.columns[~df.columns.isin(nonprotein_col_names)])
    # Then we drop all nonprotein columns 
    df.drop(columns=nonprotein_col_names, inplace=True)
    # Transform data 
    if arcsinh5_transform:
        df = np.arcsinh(df/5)
    # Make sure that the data is zero inflated 
    if is_zero_inflated:
        df = np.clip(df, a_min=0, a_max=None)
    # Then we append the curated indices to the dataframe 
    df['batch_index'] = batch_index
    df['condition_index'] = condition_index
    df['subject_index'] = subject_index
    df['cell_type_index'] = cell_type_index
    # Calculate some meta information 
    n_b = df['batch_index'].max(axis=0)+1
    n_c = df['condition_index'].max(axis=0)+1
    n_s = df['subject_index'].max(axis=0)+1
    n_types = df['cell_type_index'].max(axis=0)+1
    
    meta_dict = {
        "batch_index_col_name": batch_index_col_name,
        "condition_index_col_name": condition_index_col_name,
        "subject_index_col_name": subject_index_col_name,
        "cell_type_index_col_name": cell_type_index_col_name,
        "other_nonprotein_col_names": other_nonprotein_col_names,
        "protein_col_names": protein_col_names,
        "n_batches": n_b,
        "n_conditions": n_c,
        "n_subjects": n_s,
        "n_types": n_types,
        "N": df.shape[0],
        "y_dim": len(protein_col_names)
    }
    
    return df, meta_dict
    
    
def generate_cytoone_model_input(meta_dict: dict,
                                 x_dim: int, 
                                 n_cell_types: int,
                                 model_check_point_path: Optional[str]=None) -> dict:
    """Generate model input for CytoOne 

    By using the meta information we extracted from the data, the functions 
    constructs a dictionary as model input 
    
    Parameters
    ----------
    meta_dict : dict
        A dictionary containing meta information of the data
    x_dim : int
        Dimensions of the latent space
    n_cell_types : int
        Number of cell clusters 
    model_check_point_path : Optional[str], optional
        The path to the pretrained model. If none, random weights are used, by default None

    Returns
    -------
    dict
        A dictionary of the model input 
    """
    
    parameter_dict = {
        "y_dim": meta_dict['y_dim'],
        "x_dim": x_dim,
        "n_batches": meta_dict['n_batches'],
        "n_conditions": meta_dict['n_conditions'],
        "n_subjects": meta_dict['n_subjects'],
        "n_cell_types": n_cell_types,
        "model_check_point_path": model_check_point_path
    }
    
    return parameter_dict


def generate_pretrain_data(y_scale: float=0.001,
                           low: float=-10,
                           high: float=2.5,
                           n_sample: int=100000) -> pd.DataFrame:
    """Generate data to be used for pretraining 

    Parameters
    ----------
    y_scale : float, optional
        The scale of the normal distribution, by default 0.001
    low: float
        The lower bound 
    high: float
        The upper bound 
    n_sample : int, optional
        Number of samples, by default 100000

    Returns
    -------
    pd.DataFrame
        A dataframe containing generated pretraining data
    """
    # We first generate data frome a pretty large square 
    z = np.random.uniform(low=low, high=high, size=n_sample)
    w = np.random.uniform(low=low, high=high, size=n_sample)
    # Then we generate truly zero inflated observations 
    pure_y = np.round(1/(1+np.exp(-w))) * np.exp(z)
    # Finally, we corrupt them with a little noise 
    y = pure_y + np.random.normal(loc=0, scale=y_scale, size=n_sample)
    df = pd.DataFrame({"y": y,
                       "z": z, 
                       "w": w})
    return df