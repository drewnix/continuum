# tests/test_training.py
import pytest
from continuum.train import load_data, train_model

def test_load_data():
    data = load_data()
    assert not data.empty
    assert 'target' in data.columns

def test_train_model():
    data = load_data()
    model = train_model(data)
    assert model is not None
    assert hasattr(model, 'predict')

if __name__ == "__main__":
    pytest.main()