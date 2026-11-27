import pandas as pd
import pytest
@pytest.fixture
def df():
    return pd.read_csv("../data/processed/sms_dedup.csv")
