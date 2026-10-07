import pytest
import torch
import numpy as np
from mlye.utils import evalSinhArcsinhNormal


class TestEvalSinhArcsinhNormal:
    """Comprehensive tests for the evalSinhArcsinhNormal function."""

    @pytest.fixture
    def standard_params(self):
        """Standard parameter set for testing."""
        return {
            'loc': 0.0,
            'scale': 1.0,
            'skewness': 0.0,
            'tailweight': 1.0
        }

    @pytest.fixture
    def sample_inputs(self):
        """Standard input samples for testing."""
        return torch.randn(100)

    def test_basic_functionality(self, standard_params, sample_inputs):
        """Test basic function call and output structure."""
        z, log_qz = evalSinhArcsinhNormal(**standard_params, u=sample_inputs)
        
        assert isinstance(z, torch.Tensor)
        assert isinstance(log_qz, torch.Tensor)
        assert z.shape == sample_inputs.shape
        assert log_qz.shape == sample_inputs.shape

    def test_output_types(self, standard_params):
        """Test that outputs have correct tensor types."""
        u = torch.randn(10, dtype=torch.float32)
        z, log_qz = evalSinhArcsinhNormal(**standard_params, u=u)
        
        assert z.dtype == torch.float32
        assert log_qz.dtype == torch.float32

    def test_double_precision(self, standard_params):
        """Test with double precision tensors."""
        u = torch.randn(10, dtype=torch.float64)
        z, log_qz = evalSinhArcsinhNormal(**standard_params, u=u)
        
        assert z.dtype == torch.float64
        assert log_qz.dtype == torch.float64

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_different_batch_sizes(self, standard_params, batch_size):
        """Test function with different batch sizes."""
        u = torch.randn(batch_size)
        z, log_qz = evalSinhArcsinhNormal(**standard_params, u=u)
        
        assert z.shape == (batch_size,)
        assert log_qz.shape == (batch_size,)

    def test_multidimensional_input(self, standard_params):
        """Test function with multidimensional input tensors."""
        u = torch.randn(5, 10, 3)
        z, log_qz = evalSinhArcsinhNormal(**standard_params, u=u)
        
        assert z.shape == (5, 10, 3)
        assert log_qz.shape == (5, 10, 3)

    def test_parameter_broadcasting(self):
        """Test parameter broadcasting with different shapes."""
        u = torch.randn(5)  # Use smaller size for simpler test
        loc1 = 1.0
        loc2 = 2.0
        
        # Test that different locations produce different outputs
        z1, log_qz1 = evalSinhArcsinhNormal(loc1, 1.0, 0.0, 1.0, u)
        z2, log_qz2 = evalSinhArcsinhNormal(loc2, 1.0, 0.0, 1.0, u)
        
        assert z1.shape == u.shape
        assert z2.shape == u.shape
        assert log_qz1.shape == u.shape
        assert log_qz2.shape == u.shape
        
        # Different locations should produce different outputs
        assert not torch.allclose(z1, z2)

    def test_identity_transformation(self):
        """Test that with standard parameters and zero skewness, transformation approximates identity for small u."""
        u = torch.linspace(-0.1, 0.1, 11)
        loc = 0.0
        scale = 1.0
        skewness = 0.0
        tailweight = 1.0
        
        z, log_qz = evalSinhArcsinhNormal(loc, scale, skewness, tailweight, u)
        
        # For small u and standard parameters, z should approximate u
        torch.testing.assert_close(z, u, atol=1e-3, rtol=1e-3)

    def test_location_parameter(self):
        """Test that location parameter shifts the output correctly."""
        u = torch.randn(100)
        loc1 = 0.0
        loc2 = 5.0
        
        z1, _ = evalSinhArcsinhNormal(loc1, 1.0, 0.0, 1.0, u)
        z2, _ = evalSinhArcsinhNormal(loc2, 1.0, 0.0, 1.0, u)
        
        torch.testing.assert_close(z2 - z1, torch.full_like(u, loc2 - loc1))

    def test_scale_parameter(self):
        """Test that scale parameter scales the output correctly."""
        u = torch.randn(100)
        scale1 = 1.0
        scale2 = 2.0
        
        z1, _ = evalSinhArcsinhNormal(0.0, scale1, 0.0, 1.0, u)
        z2, _ = evalSinhArcsinhNormal(0.0, scale2, 0.0, 1.0, u)
        
        # With loc=0, scaling should be proportional
        torch.testing.assert_close(z2, z1 * scale2, atol=1e-6, rtol=1e-6)

    def test_skewness_effect(self):
        """Test that skewness parameter affects the distribution."""
        u = torch.randn(1000)
        
        z_symmetric, _ = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 1.0, u)
        z_skewed, _ = evalSinhArcsinhNormal(0.0, 1.0, 1.0, 1.0, u)
        
        # Skewed distribution should have different mean
        assert not torch.allclose(z_symmetric.mean(), z_skewed.mean(), atol=1e-2)

    def test_tailweight_effect(self):
        """Test that tailweight parameter affects the distribution."""
        u = torch.randn(1000)
        
        z_normal, _ = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 1.0, u)
        z_heavy, _ = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 2.0, u)
        
        # Different tailweights should produce different variances
        assert not torch.allclose(z_normal.var(), z_heavy.var(), atol=1e-2)

    def test_log_probability_properties(self):
        """Test properties of the log probability term."""
        u = torch.randn(100)
        _, log_qz = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 1.0, u)
        
        # Log probability should be real and finite
        assert torch.all(torch.isfinite(log_qz))
        assert torch.all(torch.isreal(log_qz))

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the function."""
        u = torch.randn(10, requires_grad=True)
        loc = torch.tensor(0.0, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        skewness = torch.tensor(0.5, requires_grad=True)
        tailweight = torch.tensor(1.5, requires_grad=True)
        
        z, log_qz = evalSinhArcsinhNormal(loc, scale, skewness, tailweight, u)
        loss = (z.sum() + log_qz.sum())
        loss.backward()
        
        # All parameters should have gradients
        assert u.grad is not None
        assert loc.grad is not None
        assert scale.grad is not None
        assert skewness.grad is not None
        assert tailweight.grad is not None

    def test_numerical_stability_extreme_u(self):
        """Test numerical stability with extreme input values."""
        u_extreme = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
        
        z, log_qz = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 1.0, u_extreme)
        
        assert torch.all(torch.isfinite(z))
        assert torch.all(torch.isfinite(log_qz))

    def test_numerical_stability_extreme_tailweight(self):
        """Test numerical stability with extreme tailweight values."""
        u = torch.randn(10)
        tailweights = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        
        for tw in tailweights:
            z, log_qz = evalSinhArcsinhNormal(0.0, 1.0, 0.0, tw.item(), u)
            assert torch.all(torch.isfinite(z))
            assert torch.all(torch.isfinite(log_qz))

    def test_edge_case_zero_scale(self):
        """Test behavior with zero scale parameter."""
        u = torch.randn(10)
        
        z, log_qz = evalSinhArcsinhNormal(5.0, 0.0, 0.0, 1.0, u)
        
        # With zero scale, all outputs should equal the location
        torch.testing.assert_close(z, torch.full_like(u, 5.0))

    def test_consistency_with_tensor_parameters(self):
        """Test consistency when parameters are tensors vs scalars."""
        u = torch.randn(10)
        
        # Scalar parameters
        z1, log_qz1 = evalSinhArcsinhNormal(1.0, 2.0, 0.5, 1.5, u)
        
        # Tensor parameters with same values
        z2, log_qz2 = evalSinhArcsinhNormal(
            torch.tensor(1.0), torch.tensor(2.0), 
            torch.tensor(0.5), torch.tensor(1.5), u
        )
        
        torch.testing.assert_close(z1, z2)
        torch.testing.assert_close(log_qz1, log_qz2)

    def test_device_consistency(self):
        """Test that function works correctly on different devices."""
        if torch.cuda.is_available():
            u_cpu = torch.randn(10)
            u_cuda = u_cpu.cuda()
            
            z_cpu, log_qz_cpu = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 1.0, u_cpu)
            z_cuda, log_qz_cuda = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 1.0, u_cuda)
            
            torch.testing.assert_close(z_cpu, z_cuda.cpu())
            torch.testing.assert_close(log_qz_cpu, log_qz_cuda.cpu())

    @pytest.mark.parametrize("loc,scale,skewness,tailweight", [
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 2.0, 0.5, 1.5),
        (-2.0, 0.5, -1.0, 2.0),
        (10.0, 3.0, 2.0, 0.8)
    ])
    def test_parametrized_combinations(self, loc, scale, skewness, tailweight):
        """Test various parameter combinations."""
        u = torch.randn(50)
        
        z, log_qz = evalSinhArcsinhNormal(loc, scale, skewness, tailweight, u)
        
        assert z.shape == u.shape
        assert log_qz.shape == u.shape
        assert torch.all(torch.isfinite(z))
        assert torch.all(torch.isfinite(log_qz))

    def test_mathematical_properties(self):
        """Test specific mathematical properties of the transformation."""
        u = torch.tensor([0.0])  # Test at u=0
        
        # At u=0, arcsinh(u) = 0, so transformation simplifies
        z, log_qz = evalSinhArcsinhNormal(2.0, 3.0, 1.0, 1.5, u)
        
        # Expected: z = 2 + 3 * sinh(1.5 * (0 + 1)) = 2 + 3 * sinh(1.5)
        expected_z = torch.tensor([2.0 + 3.0 * torch.sinh(torch.tensor(1.5)).item()])
        torch.testing.assert_close(z, expected_z, atol=1e-6, rtol=1e-6)

    def test_empty_input(self):
        """Test behavior with empty input tensor."""
        u = torch.empty(0)
        
        z, log_qz = evalSinhArcsinhNormal(0.0, 1.0, 0.0, 1.0, u)
        
        assert z.shape == (0,)
        assert log_qz.shape == (0,)
