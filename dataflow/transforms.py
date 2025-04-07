"""
Common data transformation functions that can be used as pipeline steps.
"""
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import re
import pandas as pd
import numpy as np
from dataflow.core import step


# Text transformations
@step(name="text_clean")
def clean_text(data: Union[str, List[str]], 
              remove_punctuation: bool = True,
              lowercase: bool = True) -> Union[str, List[str]]:
    """
    Clean text data by optionally removing punctuation and converting to lowercase.
    
    Args:
        data: String or list of strings to clean
        remove_punctuation: Whether to remove punctuation
        lowercase: Whether to convert to lowercase
    
    Returns:
        Cleaned text data in the same format as input
    """
    is_single_string = isinstance(data, str)
    texts = [data] if is_single_string else data
    
    result = []
    for text in texts:
        if lowercase:
            text = text.lower()
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        result.append(text)
    
    return result[0] if is_single_string else result


# Numerical transformations
@step(name="normalize")
def normalize(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize numerical data.
    
    Args:
        data: Numerical data to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Normalized data
    """
    data = np.asarray(data)
    
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(data)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std if std > 0 else np.zeros_like(data)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# DataFrame transformations
@step(name="drop_na")
def drop_na(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns and rows with too many missing values.
    
    Args:
        df: DataFrame to process
        threshold: Drop columns/rows with more than this fraction of missing values
    
    Returns:
        Processed DataFrame
    """
    # Drop columns with too many NAs
    col_threshold = int(threshold * len(df))
    df = df.dropna(axis=1, thresh=col_threshold)
    
    # Drop rows with too many NAs
    row_threshold = int(threshold * df.shape[1])
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df


@step(name="encode_categorical")
def encode_categorical(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      method: str = 'onehot',
                      drop_first: bool = False) -> pd.DataFrame:
    """
    Encode categorical variables in a DataFrame.
    
    Args:
        df: DataFrame to process
        columns: List of column names to encode (if None, detect object columns)
        method: Encoding method ('onehot', 'label', or 'ordinal')
        drop_first: Whether to drop the first category (for onehot encoding)
    
    Returns:
        DataFrame with encoded variables
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if method == 'onehot':
        for col in columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    
    elif method == 'label':
        for col in columns:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
    
    elif method == 'ordinal':
        for col in columns:
            if col in df.columns:
                categories = df[col].astype('category').cat.categories
                df[col] = df[col].astype('category').cat.codes
                
                # Store the category mappings in metadata
                if not hasattr(df, 'category_mappings'):
                    df.category_mappings = {}
                df.category_mappings[col] = dict(enumerate(categories))
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df


# Data filtering
@step(name="filter_outliers")
def filter_outliers(data: np.ndarray, method: str = 'zscore', threshold: float = 3.0) -> np.ndarray:
    """
    Filter outliers from numerical data.
    
    Args:
        data: Numerical data to process
        method: Outlier detection method ('zscore' or 'iqr')
        threshold: Threshold for outlier detection
    
    Returns:
        Filtered data with outliers removed
    """
    data = np.asarray(data)
    
    if method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return data[z_scores < threshold]
    
    elif method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


# Custom transformation factory
def apply_func(func: Callable, **kwargs) -> Callable:
    """
    Create a step that applies a custom function to data.
    
    Args:
        func: Function to apply
        **kwargs: Additional arguments to pass to the function
    
    Returns:
        Step function that applies the custom function
    """
    @step(name=f"apply_{func.__name__}")
    def apply_function(data: Any) -> Any:
        return func(data, **kwargs)
    
    return apply_function


# Splitting and sampling
@step(name="train_test_split")
def train_test_split(data: Any, test_size: float = 0.2, 
                    random_state: Optional[int] = None) -> Tuple[Any, Any]:
    """
    Split data into training and testing sets.
    
    Args:
        data: Data to split (array, DataFrame, or other indexable)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if isinstance(data, pd.DataFrame):
        indices = np.random.permutation(len(data))
        test_idx = int(test_size * len(data))
        return data.iloc[indices[test_idx:]], data.iloc[indices[:test_idx]]
    
    elif isinstance(data, np.ndarray):
        indices = np.random.permutation(len(data))
        test_idx = int(test_size * len(data))
        return data[indices[test_idx:]], data[indices[:test_idx]]
    
    else:
        # Try to handle any indexable object
        try:
            data_len = len(data)
            indices = np.random.permutation(data_len)
            test_idx = int(test_size * data_len)
            
            # Create new lists for train and test
            train = [data[i] for i in indices[test_idx:]]
            test = [data[i] for i in indices[:test_idx]]
            return train, test
        except:
            raise TypeError("Data type not supported for train_test_split")