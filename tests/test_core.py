"""
Tests for the core module of DataFlow.
"""
import pytest
import pandas as pd
import numpy as np
from dataflow.core import Pipeline, Step, step


def test_step_decorator():
    """Test that the step decorator creates a Step object."""
    @step(name="test_step")
    def add_one(x):
        return x + 1
    
    assert isinstance(add_one, Step)
    assert add_one.name == "test_step"
    assert add_one(5) == 6


def test_step_execution_time():
    """Test that step execution time is tracked."""
    @step
    def slow_step(x):
        import time
        time.sleep(0.01)  # Sleep for a short time
        return x
    
    result = slow_step(10)
    assert result == 10
    assert slow_step.execution_time > 0


def test_pipeline_creation():
    """Test creating a pipeline."""
    @step
    def add_one(x):
        return x + 1
    
    @step
    def multiply_by_two(x):
        return x * 2
    
    pipeline = Pipeline([add_one, multiply_by_two], name="test_pipeline")
    assert len(pipeline.steps) == 2
    assert pipeline.name == "test_pipeline"


def test_pipeline_execution():
    """Test executing a pipeline."""
    @step
    def add_one(x):
        return x + 1
    
    @step
    def multiply_by_two(x):
        return x * 2
    
    pipeline = Pipeline([add_one, multiply_by_two])
    result = pipeline.run(5)
    assert result == 12  # (5+1)*2 = 12


def test_pipeline_add_step():
    """Test adding steps to a pipeline."""
    @step
    def add_one(x):
        return x + 1
    
    @step
    def multiply_by_two(x):
        return x * 2
    
    pipeline = Pipeline()
    pipeline.add_step(add_one).add_step(multiply_by_two)
    
    assert len(pipeline.steps) == 2
    result = pipeline.run(5)
    assert result == 12


def test_pipeline_composition():
    """Test composing pipelines with the | operator."""
    @step
    def add_one(x):
        return x + 1
    
    @step
    def multiply_by_two(x):
        return x * 2
    
    @step
    def square(x):
        return x ** 2
    
    pipeline1 = Pipeline([add_one, multiply_by_two])
    pipeline2 = Pipeline([square])
    
    combined = pipeline1 | pipeline2
    
    assert len(combined.steps) == 3
    result = combined.run(5)
    assert result == 144  # ((5+1)*2)^2 = 12^2 = 144


def test_step_caching():
    """Test that step caching works correctly."""
    call_count = 0
    
    @step(cache=True)
    def counting_step(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # First call should execute the function
    result1 = counting_step(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call with same input should use cache
    result2 = counting_step(5)
    assert result2 == 10
    assert call_count == 1  # Count should not increase
    
    # Call with different input should execute again
    result3 = counting_step(10)
    assert result3 == 20
    assert call_count == 2
    
    # Clear cache and call again
    counting_step.clear_cache()
    result4 = counting_step(5)
    assert result4 == 10
    assert call_count == 3


def test_pipeline_execution_stats():
    """Test that pipeline execution stats are recorded."""
    @step
    def step1(x):
        return x + 1
    
    @step
    def step2(x):
        return x * 2
    
    pipeline = Pipeline([step1, step2])
    result = pipeline.run(5)
    
    stats = pipeline.execution_stats
    assert 'total_time' in stats
    assert 'steps' in stats
    assert step1.name in stats['steps']
    assert step2.name in stats['steps']
    assert 'execution_time' in stats['steps'][step1.name]


def test_step_validation():
    """Test step validation."""
    def validate_positive(x):
        return x > 0, "Value must be positive" if x <= 0 else ""
    
    # Use validate_input=True to validate the input value, not the output
    @step(validate=validate_positive, validate_input=True)
    def square_root(x):
        return x ** 0.5
    
    # Valid input
    result = square_root(4)
    assert result == 2.0
    
    # Invalid input should raise ValueError
    with pytest.raises(ValueError):
        square_root(-4)


def test_step_output_validation():
    """Test validation of step outputs."""
    def validate_even(x):
        return x % 2 == 0, "Output must be even" if x % 2 != 0 else ""
    
    # Use validate_input=False to validate the output value
    @step(validate=validate_even, validate_input=False)
    def double(x):
        return x * 2
    
    # Should pass validation (output is even)
    result = double(5)
    assert result == 10
    
    # Test a function that would fail output validation
    @step(validate=validate_even, validate_input=False)
    def add_one(x):
        return x + 1
    
    # Should fail validation when given odd number
    with pytest.raises(ValueError):
        add_one(1)  # 1+1=2, which is even
        add_one(2)  # 2+1=3, which is odd, so this will fail


def test_pandas_dataframe_pipeline():
    """Test pipeline with pandas DataFrame."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    @step
    def add_column_c(data):
        data = data.copy()
        data['C'] = data['A'] + data['B']
        return data
    
    @step
    def multiply_column_a(data):
        data = data.copy()
        data['A'] = data['A'] * 2
        return data
    
    pipeline = Pipeline([add_column_c, multiply_column_a])
    result = pipeline.run(df)
    
    assert 'C' in result.columns
    assert result['A'].tolist() == [2, 4, 6]
    assert result['C'].tolist() == [5, 7, 9]