"""
Example demonstrating DataFlow for data cleaning tasks.
"""

import pandas as pd
import numpy as np
from dataflow.core import Pipeline, step
from dataflow.validators import has_columns, no_missing_values, value_range
import dataflow.utils as utils

# Create a messy dataset
np.random.seed(42)  # For reproducibility
data = pd.DataFrame({
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 90, size=100),
    'income': np.random.normal(60000, 15000, size=100),
    'signup_date': pd.date_range('2022-01-01', periods=100, freq='D'),
    'last_purchase': pd.date_range('2022-03-01', periods=100, freq='D'),
    'region': np.random.choice(['North', 'South', 'East', 'West', None], 100, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    'satisfaction': np.random.choice([1, 2, 3, 4, 5, None], 100, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),
})

# Make some data messy
# Add some extreme outliers to income
data.loc[10, 'income'] = -5000  # Negative income
data.loc[20, 'income'] = 1000000  # Very high income

# Add some invalid ages
data.loc[30, 'age'] = 5  # Too young
data.loc[40, 'age'] = 120  # Too old
data.loc[50, 'age'] = np.nan  # Missing age

# Add some date formatting issues (create strings instead of dates for some rows)
data.loc[60, 'signup_date'] = 'Jan 1, 2022'
data.loc[70, 'last_purchase'] = '2022-03-70'  # Invalid date

print("Original Data Sample:")
print(data.head())
print(f"\nShape: {data.shape}")
print(f"Missing values:\n{data.isna().sum()}")
print(f"Data types:\n{data.dtypes}")
print("\n" + "="*50 + "\n")

# Define our cleaning steps

@step(name="fix_dates", validate=has_columns(['signup_date', 'last_purchase']))
def fix_date_format(df):
    """Convert string dates to datetime format."""
    df = df.copy()
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_purchase'] = pd.to_datetime(df['last_purchase'], errors='coerce')
    return df

@step(name="handle_missing_values")
def handle_missing_values(df):
    """Handle missing values in each column appropriately."""
    df = df.copy()
    
    # Fill missing regions with 'Unknown'
    df['region'] = df['region'].fillna('Unknown')
    
    # Fill missing ages with median age
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age)
    
    # Fill missing satisfaction with median
    median_satisfaction = df['satisfaction'].median()
    df['satisfaction'] = df['satisfaction'].fillna(median_satisfaction)
    
    return df

@step(name="fix_age_outliers", validate=value_range('age', min_val=18, max_val=100))
def fix_age_outliers(df):
    """Fix outliers in age column."""
    df = df.copy()
    
    # Cap age between 18 and 100
    df['age'] = df['age'].clip(lower=18, upper=100)
    
    return df

@step(name="fix_income_outliers")
def fix_income_outliers(df):
    """Fix outliers in income column."""
    df = df.copy()
    
    # Replace negative income with 0
    df.loc[df['income'] < 0, 'income'] = 0
    
    # Cap high incomes (use 3 standard deviations as a rule of thumb)
    mean = df['income'].mean()
    std = df['income'].std()
    upper_limit = mean + 3 * std
    
    df.loc[df['income'] > upper_limit, 'income'] = upper_limit
    
    return df

@step(name="create_features")
def create_features(df):
    """Create useful derived features."""
    df = df.copy()
    
    # Calculate customer tenure in days
    df['tenure_days'] = (df['last_purchase'] - df['signup_date']).dt.days
    
    # Create income brackets
    conditions = [
        (df['income'] < 30000),
        (df['income'] >= 30000) & (df['income'] < 60000),
        (df['income'] >= 60000) & (df['income'] < 90000),
        (df['income'] >= 90000)
    ]
    
    choices = ['Low', 'Medium', 'High', 'Very High']
    df['income_bracket'] = np.select(conditions, choices, default='Unknown')
    
    # Create age groups
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[18, 25, 35, 50, 65, 100],
        labels=['18-24', '25-34', '35-49', '50-64', '65+']
    )
    
    return df

@step(name="fix_dtypes")
def fix_data_types(df):
    """Ensure correct data types."""
    df = df.copy()
    
    # Convert customer_id to string
    df['customer_id'] = df['customer_id'].astype(str)
    
    # Convert age to int
    df['age'] = df['age'].astype(int)
    
    # Convert satisfaction to int
    df['satisfaction'] = df['satisfaction'].astype(int)
    
    return df

# Create our pipeline
cleaning_pipeline = Pipeline(name="customer_data_cleaning")
cleaning_pipeline.add_step(fix_date_format) \
                .add_step(handle_missing_values) \
                .add_step(fix_age_outliers) \
                .add_step(fix_income_outliers) \
                .add_step(create_features) \
                .add_step(fix_data_types)

# Run the pipeline
try:
    clean_data = cleaning_pipeline.run(data)
    
    print("Cleaned Data Sample:")
    print(clean_data.head())
    print(f"\nShape: {clean_data.shape}")
    print(f"Missing values:\n{clean_data.isna().sum()}")
    print(f"Data types:\n{clean_data.dtypes}")
    
    # Print execution statistics
    print("\nPipeline Execution Statistics:")
    for step_name, details in cleaning_pipeline.execution_stats['steps'].items():
        print(f"{step_name}: {details['execution_time']*1000:.2f}ms")
    print(f"Total time: {cleaning_pipeline.execution_stats['total_time']*1000:.2f}ms")
    
    # Visualize the pipeline
    print("\nPipeline Visualization:")
    print(utils.create_dag_visualization(cleaning_pipeline))
    
    # Save the clean data to CSV (optional)
    # clean_data.to_csv('clean_customer_data.csv', index=False)
    
    # Save the pipeline for future use (optional)
    # utils.save_pipeline(cleaning_pipeline, 'cleaning_pipeline.pkl')
    
except Exception as e:
    print(f"Error in pipeline: {e}")