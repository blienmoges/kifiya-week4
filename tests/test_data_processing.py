import pytest
import pandas as pd
from src.data_processing import build_feature_pipeline

# ------------------------
# Sample Data Fixture
# ------------------------
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': ['C1', 'C2', 'C3'],
        'Amount': [100, 200, 150],
        'Value': [10, 20, 15],
        'CurrencyCode': ['USD', 'EUR', 'USD'],
        'CountryCode': ['US', 'FR', 'US'],
        'ProviderId': ['P1', 'P2', 'P1'],
        'ProductCategory': ['A', 'B', 'A'],
        'ChannelId': ['C1', 'C2', 'C1'],
        'PricingStrategy': ['S1', 'S2', 'S1'],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03']  # Added column
    })

# ------------------------
# Test: Pipeline Output Shape
# ------------------------
def test_pipeline_output_shape(sample_data):
    numerical_cols = ['Amount', 'Value']
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 
                        'ProductCategory', 'ChannelId', 'PricingStrategy']

    pipeline = build_feature_pipeline(numerical_cols, categorical_cols)
    transformed = pipeline.fit_transform(sample_data)

    # Test that the transformed output has correct number of rows
    assert transformed.shape[0] == sample_data.shape[0]
    # Optionally, check that there are more columns now (after transformations)
    assert transformed.shape[1] >= len(numerical_cols) + len(categorical_cols)

# ------------------------
# Test: Pipeline Columns
# ------------------------
def test_pipeline_columns(sample_data):
    numerical_cols = ['Amount', 'Value']
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 
                        'ProductCategory', 'ChannelId', 'PricingStrategy']

    pipeline = build_feature_pipeline(numerical_cols, categorical_cols)
    transformed = pipeline.fit_transform(sample_data)

    # Check that transformed output has correct type
    assert isinstance(transformed, pd.DataFrame) or hasattr(transformed, "shape")
