"""
A simple example of using DataFlow to process data.
"""

import pandas as pd
import numpy as np
from dataflow.core import Pipeline, step
import dataflow.utils as utils

# Create some sample data
data = pd.DataFrame({
    'id': range(1, 11),
    'value': np.random.rand(10) * 100,
    'category': np.random.choice(['A', 'B', 'C'], 10),
    'date': pd.date_range('2023-01-01', periods=10, freq='D'),
})

# Intentionally add some missing values
data.loc[3, 'value'] = None
data.loc[7, 'category'] = None

print("Original data:")
print(data)
print()

# Define some transformation steps
@step(name="clean_missing")
def clean_missing_values(df):
    df = df.copy()
    # Fill missing values in 'value' column with mean
    df['value'] = df['value'].fillna(df['value'].mean())
    # Fill missing values in 'category' column with 'Unknown'
    df['category'] = df['category'].fillna('Unknown')
    return df

@step(name="add_features")
def add_features(df):
    df = df.copy()
    # Extract day of week
    df['day_of_week'] = df['date'].dt.day_name()
    # Create a derived value
    df['value_squared'] = df['value'] ** 2
    return df

@step(name="filter_high_values")
def filter_high_values(df, threshold=50):
    return df[df['value'] > threshold]

# Create and execute the pipeline
pipeline = Pipeline(name="data_processing")
pipeline.add_step(clean_missing_values) \
        .add_step(add_features) \
        .add_step(filter_high_values, threshold=60)

# Run the pipeline and get the result
result = pipeline.run(data)

# Print results
print("Processed data:")
print(result)
print()

# Print execution statistics
print("Pipeline execution statistics:")
for step_name, details in pipeline.execution_stats['steps'].items():
    print(f"{step_name}: {details['execution_time']*1000:.2f}ms")
print(f"Total time: {pipeline.execution_stats['total_time']*1000:.2f}ms")
print()

# Visualize the pipeline
print("Pipeline visualization:")
print(utils.create_dag_visualization(pipeline))