"""
Shared pytest configuration and fixtures for mlye tests
"""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def device():
    """Return the appropriate device for testing"""
    return torch.device('cpu')  # Use CPU for tests for consistency


@pytest.fixture
def small_data():
    """Generate small dataset for quick tests"""
    torch.manual_seed(42)
    batch_size, T = 20, 4
    y = torch.randn(batch_size, T) * 0.3
    z = torch.randn(batch_size, T) * 0.2
    return y, z


@pytest.fixture
def medium_data():
    """Generate medium dataset for standard tests"""
    torch.manual_seed(42)
    batch_size, T = 100, 6
    y = torch.randn(batch_size, T) * 0.5
    z = torch.randn(batch_size, T) * 0.3
    return y, z


@pytest.fixture
def large_data():
    """Generate large dataset for stress tests"""
    torch.manual_seed(42)
    batch_size, T = 500, 10
    y = torch.randn(batch_size, T) * 0.4
    z = torch.randn(batch_size, T) * 0.25
    return y, z


@pytest.fixture(params=[1, 3, 6, 10])
def time_periods(request):
    """Parametrize tests across different numbers of time periods"""
    return request.param


@pytest.fixture(params=[10, 50, 200])
def batch_sizes(request):
    """Parametrize tests across different batch sizes"""
    return request.param


@pytest.fixture(params=['normal', 'student_t'])
def distribution_types(request):
    """Parametrize tests across distribution types"""
    return request.param