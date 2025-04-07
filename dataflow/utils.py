"""
Utility functions for DataFlow pipelines.
"""
from typing import Any, Dict, List, Optional, Union
import json
import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from dataflow.core import Pipeline, Step


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """
    Save a pipeline to a file using pickle.
    
    Args:
        pipeline: Pipeline to save
        filepath: Path to save the pipeline to
    """
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)


def load_pipeline(filepath: str) -> Pipeline:
    """
    Load a pipeline from a file.
    
    Args:
        filepath: Path to load the pipeline from
        
    Returns:
        Loaded pipeline
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def pipeline_summary(pipeline: Pipeline) -> Dict:
    """
    Generate a summary of a pipeline.
    
    Args:
        pipeline: Pipeline to summarize
        
    Returns:
        Dictionary with pipeline summary information
    """
    return {
        "name": pipeline.name,
        "steps": [step.name for step in pipeline.steps],
        "num_steps": len(pipeline.steps),
        "execution_stats": pipeline.execution_stats
    }


def export_pipeline_metrics(pipeline: Pipeline, filepath: str) -> None:
    """
    Export pipeline execution metrics to a JSON file.
    
    Args:
        pipeline: Pipeline with execution metrics
        filepath: Path to save the metrics to
    """
    metrics = {
        "pipeline_name": pipeline.name,
        "total_execution_time": pipeline.execution_stats.get('total_time', 0),
        "step_metrics": pipeline.execution_stats.get('steps', {}),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def describe_data(data: Any) -> Dict:
    """
    Generate a description of data.
    
    Args:
        data: Data to describe
        
    Returns:
        Dictionary with data description
    """
    result = {
        "type": type(data).__name__,
    }
    
    if isinstance(data, pd.DataFrame):
        result.update({
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "missing_values": data.isna().sum().to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        })
    
    elif isinstance(data, pd.Series):
        result.update({
            "length": len(data),
            "dtype": str(data.dtype),
            "missing_values": data.isna().sum(),
            "memory_usage": data.memory_usage(deep=True) / (1024 * 1024),  # MB
        })
    
    elif isinstance(data, np.ndarray):
        result.update({
            "shape": data.shape,
            "dtype": str(data.dtype),
            "memory_usage": data.nbytes / (1024 * 1024),  # MB
            "missing_values": np.isnan(data).sum() if np.issubdtype(data.dtype, np.number) else None
        })
    
    elif isinstance(data, (list, tuple)):
        result.update({
            "length": len(data),
            "element_types": list(set(type(item).__name__ for item in data[:10])) if data else []
        })
    
    elif isinstance(data, dict):
        result.update({
            "length": len(data),
            "key_types": list(set(type(k).__name__ for k in data.keys()))[:5],
            "value_types": list(set(type(v).__name__ for v in data.values()))[:5]
        })
    
    return result


def log_pipeline_execution(pipeline: Pipeline, log_file: str) -> None:
    """
    Append pipeline execution log to a log file.
    
    Args:
        pipeline: Executed pipeline with stats
        log_file: Path to log file
    """
    timestamp = datetime.now().isoformat()
    stats = pipeline.execution_stats
    
    log_entry = {
        "timestamp": timestamp,
        "pipeline_name": pipeline.name,
        "total_time": stats.get('total_time', 0),
        "steps_executed": len(stats.get('steps', {})),
        "step_times": {name: details.get('execution_time', 0) for name, details in stats.get('steps', {}).items()}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def create_dag_visualization(pipeline: Pipeline, output_path: Optional[str] = None) -> str:
    """
    Create a simple text-based DAG visualization of a pipeline.
    
    Args:
        pipeline: Pipeline to visualize
        output_path: Optional path to save visualization to
        
    Returns:
        String with the visualization
    """
    if not pipeline.steps:
        viz = "Empty Pipeline"
        if output_path:
            with open(output_path, 'w') as f:
                f.write(viz)
        return viz
    
    lines = [f"Pipeline: {pipeline.name}", ""]
    
    # Create the DAG visualization
    for i, step in enumerate(pipeline.steps):
        # Add step with index
        prefix = "└── " if i == len(pipeline.steps) - 1 else "├── "
        lines.append(f"{prefix}{i+1}. {step.name}")
        
        # Add execution time if available
        if pipeline.execution_stats and 'steps' in pipeline.execution_stats:
            step_stats = pipeline.execution_stats['steps'].get(step.name, {})
            if 'execution_time' in step_stats:
                time_ms = step_stats['execution_time'] * 1000
                time_str = f"{time_ms:.2f}ms"
                indent = "    " if i == len(pipeline.steps) - 1 else "│   "
                lines.append(f"{indent}└── Time: {time_str}")
    
    # Add total time if available
    if pipeline.execution_stats and 'total_time' in pipeline.execution_stats:
        total_time_ms = pipeline.execution_stats['total_time'] * 1000
        lines.append("")
        lines.append(f"Total execution time: {total_time_ms:.2f}ms")
    
    viz = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(viz)
            
    return viz