"""
Validation functions for pipeline steps to ensure data quality.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


def is_dataframe(data: Any) -> Tuple[bool, str]:
    """Validate that data is a pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        return False, "Expected a pandas DataFrame"
    return True, ""


def has_columns(columns: List[str]) -> Callable:
    """
    Create a validator that checks if a DataFrame has the required columns.
    
    Args:
        columns: List of column names that must be present
        
    Returns:
        Validator function
    """
    def validator(data: Any) -> Tuple[bool, str]:
        if not isinstance(data, pd.DataFrame):
            return False, "Expected a pandas DataFrame"
        
        missing = set(columns) - set(data.columns)
        if missing:
            return False, f"Missing required columns: {missing}"
        
        return True, ""
    
    return validator


def no_missing_values(columns: Optional[List[str]] = None) -> Callable:
    """
    Create a validator that checks for missing values in a DataFrame.
    
    Args:
        columns: Optional list of columns to check (if None, check all columns)
        
    Returns:
        Validator function
    """
    def validator(data: Any) -> Tuple[bool, str]:
        if not isinstance(data, pd.DataFrame):
            return False, "Expected a pandas DataFrame"
        
        cols_to_check = columns if columns is not None else data.columns
        cols_to_check = [col for col in cols_to_check if col in data.columns]
        
        missing_counts = data[cols_to_check].isna().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if not missing_cols.empty:
            msg = "Missing values found in columns: " + ", ".join(
                f"{col} ({count})" for col, count in missing_cols.items()
            )
            return False, msg
        
        return True, ""
    
    return validator


def value_range(column: str, min_val: Optional[float] = None, 
               max_val: Optional[float] = None) -> Callable:
    """
    Create a validator that checks if values in a column are within a specified range.
    
    Args:
        column: Column to check
        min_val: Minimum allowed value (if None, no minimum check)
        max_val: Maximum allowed value (if None, no maximum check)
        
    Returns:
        Validator function
    """
    def validator(data: Any) -> Tuple[bool, str]:
        if not isinstance(data, pd.DataFrame):
            return False, "Expected a pandas DataFrame"
        
        if column not in data.columns:
            return False, f"Column '{column}' not found"
        
        if min_val is not None:
            below_min = data[column] < min_val
            if below_min.any():
                count = below_min.sum()
                return False, f"{count} values in '{column}' below minimum ({min_val})"
        
        if max_val is not None:
            above_max = data[column] > max_val
            if above_max.any():
                count = above_max.sum()
                return False, f"{count} values in '{column}' above maximum ({max_val})"
        
        return True, ""
    
    return validator


def unique_values(column: str) -> Callable:
    """
    Create a validator that checks if a column has only unique values.
    
    Args:
        column: Column to check
        
    Returns:
        Validator function
    """
    def validator(data: Any) -> Tuple[bool, str]:
        if not isinstance(data, pd.DataFrame):
            return False, "Expected a pandas DataFrame"
        
        if column not in data.columns:
            return False, f"Column '{column}' not found"
        
        if data[column].duplicated().any():
            dup_count = data[column].duplicated().sum()
            return False, f"Found {dup_count} duplicate values in column '{column}'"
        
        return True, ""
    
    return validator


def column_type(column: str, dtype: Union[str, type]) -> Callable:
    """
    Create a validator that checks if a column has the expected data type.
    
    Args:
        column: Column to check
        dtype: Expected data type
        
    Returns:
        Validator function
    """
    def validator(data: Any) -> Tuple[bool, str]:
        if not isinstance(data, pd.DataFrame):
            return False, "Expected a pandas DataFrame"
        
        if column not in data.columns:
            return False, f"Column '{column}' not found"
        
        current_type = data[column].dtype
        if not np.issubdtype(current_type, dtype):
            return False, f"Column '{column}' has type {current_type}, expected {dtype}"
        
        return True, ""
    
    return validator


def row_count(min_rows: Optional[int] = None, max_rows: Optional[int] = None) -> Callable:
    """
    Create a validator that checks if the data has an acceptable number of rows.
    
    Args:
        min_rows: Minimum number of rows required (if None, no minimum check)
        max_rows: Maximum number of rows allowed (if None, no maximum check)
        
    Returns:
        Validator function
    """
    def validator(data: Any) -> Tuple[bool, str]:
        if isinstance(data, pd.DataFrame):
            row_count = len(data)
        elif isinstance(data, np.ndarray):
            row_count = data.shape[0]
        else:
            try:
                row_count = len(data)
            except:
                return False, "Cannot determine row count for this data type"
        
        if min_rows is not None and row_count < min_rows:
            return False, f"Data has {row_count} rows, minimum required is {min_rows}"
        
        if max_rows is not None and row_count > max_rows:
            return False, f"Data has {row_count} rows, maximum allowed is {max_rows}"
        
        return True, ""
    
    return validator


def combine_validators(*validators: Callable) -> Callable:
    """
    Combine multiple validators into a single validator.
    
    Args:
        *validators: Validator functions to combine
        
    Returns:
        Combined validator function
    """
    def combined_validator(data: Any) -> Tuple[bool, str]:
        for validator in validators:
            is_valid, error_msg = validator(data)
            if not is_valid:
                return False, error_msg
        
        return True, ""
    
    return combined_validator