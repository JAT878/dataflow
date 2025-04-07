# DataFlow

**DataFlow** is a lightweight Python library for building composable, maintainable data processing pipelines with minimal code.

[![PyPI version](https://img.shields.io/badge/pypi-0.1.0-blue.svg)](https://pypi.org/project/dataflow-pipeline/)
[![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/dataflow-pipeline/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Elegant pipelines**: Build data flows with clean, readable syntax
- **Composable**: Mix and match steps, combine pipelines
- **Debuggable**: Detailed execution statistics and visualization
- **Validated**: Optional validation between pipeline steps
- **Extensible**: Easy to add custom transformations
- **Lightweight**: Minimal dependencies with focus on performance

## 🚀 Installation

```bash
pip install dataflow-pipeline
```

## 🔍 Overview

DataFlow helps you construct data processing pipelines that are:

- **Easy to read and reason about**
- **Reusable across projects**
- **Self-documenting** with execution statistics
- **Simple to debug** when things go wrong

Whether you're cleaning messy data, preparing features for ML models, or building ETL processes, DataFlow makes your code more maintainable and your processes more reliable.

## 🏁 Quick Start

```python
import pandas as pd
from dataflow.core import Pipeline, step
from dataflow.transforms import drop_na, encode_categorical

# Load data
df = pd.read_csv("customer_data.csv")

# Create a pipeline
pipeline = Pipeline(name="data_cleaning")

# Define a custom step
@step(name="fix_dates", cache=True)
def format_dates(data):
    data = data.copy()
    data["signup_date"] = pd.to_datetime(data["signup_date"], errors="coerce")
    return data

# Build pipeline with method chaining
pipeline.add_step(format_dates) \
        .add_step(drop_na) \
        .add_step(encode_categorical, columns=["category", "region"])

# Or use the | operator for more readable pipeline construction
from dataflow.transforms import normalize
pipeline = pipeline | normalize

# Execute the pipeline
result = pipeline.run(df)

# View execution statistics
print(pipeline.execution_stats)
```

## ✨ Key Concepts

### Pipeline

A `Pipeline` is a sequence of data processing steps that can be executed on input data:

```python
from dataflow.core import Pipeline

# Create empty pipeline
pipeline = Pipeline(name="my_pipeline")

# Add steps and execute
result = pipeline.add_step(step1).add_step(step2).run(data)
```

### Step

A `Step` is a single data transformation. Create steps with the `@step` decorator:

```python
from dataflow.core import step

@step(name="my_transformer", cache=True)
def transform_data(data, param1=default1):
    # Transform data
    return transformed_data
```

### Pipeline Composition

Pipelines can be composed using the `|` operator:

```python
# Create separate pipelines for different stages
cleaning = Pipeline([clean_step1, clean_step2])
feature_eng = Pipeline([feature_step1, feature_step2])
modeling = Pipeline([model_step])

# Combine them
full_pipeline = cleaning | feature_eng | modeling

# Execute the full pipeline
result = full_pipeline.run(raw_data)
```

### Validation

Add validation between steps to catch issues early:

```python
from dataflow.validators import has_columns, no_missing_values

# Validate that required columns exist and have no missing values
step = normalize_data(
    validate=has_columns(["age", "income", "location"])
)
```

### Execution Statistics

Track performance and identify bottlenecks:

```python
# After running a pipeline
stats = pipeline.execution_stats

# Print execution times for each step
for step_name, details in stats['steps'].items():
    print(f"{step_name}: {details['execution_time']*1000:.2f}ms")

# Get total execution time
print(f"Total time: {stats['total_time']*1000:.2f}ms")
```

### Visualization

Visualize your pipeline structure:

```python
import dataflow.utils as utils

# Generate a text-based DAG visualization
viz = utils.create_dag_visualization(pipeline)
print(viz)
```

## 🛠️ Common Use Cases

### Data Cleaning

```python
from dataflow.core import Pipeline
from dataflow.transforms import drop_na, encode_categorical

cleaning = Pipeline(name="data_cleaning")
cleaning.add_step(drop_na) \
       .add_step(encode_categorical) \
       .add_step(custom_cleaning_function)
```

### ML Feature Engineering

```python
from dataflow.transforms import normalize

features = Pipeline(name="feature_engineering")
features.add_step(create_derived_features) \
       .add_step(normalize, method='zscore') \
       .add_step(feature_selection)
```

### Complete ML Workflow

```python
# Define pipeline stages
preprocessing = Pipeline([...])  # Load, clean data
feature_eng = Pipeline([...])    # Create features
model_train = Pipeline([...])    # Train and evaluate model

# Connect everything
ml_pipeline = preprocessing | feature_eng | model_train

# Run the entire workflow
results = ml_pipeline.run(raw_data)
```

## 📊 Real-World Example

This example shows a complete ML workflow using the Titanic dataset:

```python
import pandas as pd
from dataflow.core import Pipeline, step
from dataflow.transforms import drop_na, encode_categorical, train_test_split
from dataflow.validators import has_columns

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Feature engineering steps
@step(name="extract_title")
def extract_title(data):
    data = data.copy()
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    title_mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Rare",
        "Countess": "Rare", "Ms": "Miss", "Lady": "Rare", "Jonkheer": "Rare",
        "Don": "Rare", "Dona": "Rare", "Mme": "Rare", "Capt": "Rare", "Sir": "Rare"
    }
    data['Title'] = data['Title'].map(lambda x: title_mapping.get(x, 'Rare'))
    return data

@step(name="fill_missing_age")
def fill_missing_age(data):
    data = data.copy()
    # Group by Sex and Pclass and fill with median age
    data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median()))
    return data

# Model training step
@step(name="train_model")
def train_model(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    # Initialize model and train
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }

# Build the pipeline
ml_pipeline = Pipeline(name="titanic_survival_prediction")
ml_pipeline.add_step(extract_title) \
          .add_step(fill_missing_age) \
          .add_step(drop_na) \
          .add_step(encode_categorical, validate=has_columns(['Sex', 'Embarked', 'Title'])) \
          .add_step(lambda data: (data.drop('Survived', axis=1), data['Survived'])) \
          .add_step(train_test_split, test_size=0.2) \
          .add_step(train_model)

# Execute
result = ml_pipeline.run(df)
print(f"Model accuracy: {result['accuracy']:.4f}")
```

## 🔄 Why DataFlow?

### Without DataFlow:

```python
# Traditional approach - hard to read, test and reuse
def process_data(df):
    # Step 1: Clean data
    df = df.dropna()
    
    # Step 2: Transform dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Step 3: Encode categoricals
    df = pd.get_dummies(df, columns=['category'])
    
    # Step 4: Normalize
    df['amount'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
    return df

result = process_data(df)
```

### With DataFlow:

```python
# DataFlow approach - modular, testable, composable
from dataflow.core import Pipeline, step
from dataflow.transforms import drop_na, encode_categorical

@step(name="format_dates")
def format_dates(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    return data

@step(name="normalize_amount")
def normalize_amount(data):
    data = data.copy()
    data['amount'] = (data['amount'] - data['amount'].mean()) / data['amount'].std()
    return data

pipeline = Pipeline(name="data_processing")
pipeline.add_step(drop_na) \
        .add_step(format_dates) \
        .add_step(encode_categorical, columns=['category']) \
        .add_step(normalize_amount)

result = pipeline.run(df)
```

## 💡 Benefits

- **Modular design**: Each step has a single responsibility
- **Self-documenting**: Pipeline structure clearly shows data flow
- **Reusable components**: Steps can be reused across projects
- **Performance insights**: Track execution time of each step
- **Error handling**: Validate data between steps
- **Debugging**: Easy to inspect data at each stage

## 📚 API Documentation

For complete API documentation, visit [the official docs](https://dataflow-pipeline.readthedocs.io/).

## 📝 License

MIT License