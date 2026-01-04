import pytest
import pandas as pd
import os

def test_data_file_exists():
    """Check if the fashion sales data file exists"""
    assert os.path.exists('fashion_sales_data.csv'), "Data file not found!"

def test_data_loads_correctly():
    """Check if CSV loads without errors and has expected columns"""
    df = pd.read_csv('fashion_sales_data.csv')
    assert not df.empty, "Data file is empty!"
    
    # Check for critical columns (adjust based on your actual CSV structure)
    expected_columns = ['date', 'sales']  # Modify based on your actual columns
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

def test_data_has_no_nulls_in_critical_columns():
    """Check that critical columns don't have null values"""
    df = pd.read_csv('fashion_sales_data.csv')
    
    # Adjust based on your critical columns
    critical_columns = ['sales']  # Modify based on your needs
    
    for col in critical_columns:
        if col in df.columns:
            assert df[col].notna().all(), f"Column {col} has null values!"

def test_sales_values_are_positive():
    """Check that sales values are non-negative"""
    df = pd.read_csv('fashion_sales_data.csv')
    
    if 'sales' in df.columns:
        assert (df['sales'] >= 0).all(), "Negative sales values found!"

def test_date_column_is_parseable():
    """Check that date column can be parsed"""
    df = pd.read_csv('fashion_sales_data.csv')
    
    if 'date' in df.columns:
        try:
            pd.to_datetime(df['date'])
        except Exception as e:
            pytest.fail(f"Date column cannot be parsed: {e}")

# Optional: Test for newsvendor calculation logic
def test_newsvendor_calculation():
    """Test the basic newsvendor model calculation"""
    cost = 10
    price = 20
    overage_cost = cost
    underage_cost = price - cost
    
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    
    assert 0 < critical_ratio < 1, "Critical ratio should be between 0 and 1"
    assert critical_ratio == 0.5, "For this example, critical ratio should be 0.5"
