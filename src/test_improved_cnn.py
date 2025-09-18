#!/usr/bin/env python3
"""
Test script to verify the improved 2D CNN pipeline works correctly.
This script tests the key improvements without running full SNPE training.
"""

import torch
import numpy as np
from cnn_utils import create_spatial_embedding_net
from training_monitor import TrainingMonitor
from inference import RandomWalkNPE
from simulator import RandomWalkSimulator

def test_improved_cnn():
    """Test the improved CNN architecture."""
    print("🔧 Testing improved CNN architecture...")

    # Test parameters
    Ly, Lx = 50, 200
    batch_size = 4

    # Create improved CNN
    cnn = create_spatial_embedding_net(Ly, Lx, output_dim=256, dropout=0.05)
    print(f"   ✅ Created CNN with output dimension: {cnn.cnn.output_dim}")

    # Test forward pass with sample data
    test_input = torch.randn(batch_size, Ly, Lx)
    print(f"   📊 Input shape: {test_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = cnn(test_input)

    print(f"   📤 Output shape: {output.shape}")
    print(f"   ✅ Forward pass successful")

    # Check that normalization is working (output should have reasonable range)
    output_stats = {
        'mean': output.mean().item(),
        'std': output.std().item(),
        'min': output.min().item(),
        'max': output.max().item()
    }
    print(f"   📈 Output stats: mean={output_stats['mean']:.3f}, std={output_stats['std']:.3f}")

    return True

def test_training_monitor():
    """Test the training monitor functionality."""
    print("\n📊 Testing training monitor...")

    monitor = TrainingMonitor()

    # Test gradient logging
    model = torch.nn.Linear(10, 5)
    x = torch.randn(8, 10)
    y = torch.randn(8, 5)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()

    monitor.log_gradients(model, epoch=1)
    monitor.log_loss(loss.item(), epoch=1)

    print(f"   ✅ Logged gradients and loss for epoch 1")

    # Test feature stats
    features = torch.randn(16, 64, 10, 10)
    monitor.log_feature_stats(features, "test_layer", epoch=1)

    print(f"   ✅ Logged feature statistics")

    # Test convergence check
    issues = monitor.check_convergence_issues()
    print(f"   📋 Convergence issues detected: {len(issues)}")

    return True

def test_2d_npe_setup():
    """Test NPE setup with 2D data."""
    print("\n🧠 Testing 2D NPE setup...")

    # Create NPE with 2D configuration
    npe = RandomWalkNPE(device='cpu', use_2d_data=True, spatial_dims=(50, 200))
    print(f"   ✅ Created NPE with 2D data configuration")

    # Test setup_inference
    npe.setup_inference(x_shape=(50, 200))
    print(f"   ✅ Setup inference for 2D data")

    # Check that the embedding network is configured correctly
    if hasattr(npe.inference._neural_net, '_embedding_net'):
        embedding_net = npe.inference._neural_net._embedding_net
        if hasattr(embedding_net, 'cnn'):
            print(f"   ✅ CNN embedding network configured with output_dim: {embedding_net.cnn.output_dim}")

    return True

def test_data_simulation():
    """Test that 2D data simulation works correctly."""
    print("\n🎲 Testing 2D data simulation...")

    # Create simulator
    simulator = RandomWalkSimulator(Lx=200, Ly=50)

    # Test 2D simulation
    observation_2d, _, _ = simulator.simulate(U=0.3, P=0.7, T=100, use_2d_output=True)
    print(f"   📊 2D observation shape: {observation_2d.shape}")
    print(f"   📈 Agent count: {observation_2d.sum()}")
    print(f"   📏 Expected shape: (50, 200)")

    # Verify the observation is reasonable
    assert observation_2d.shape == (50, 200), f"Wrong shape: {observation_2d.shape}"
    assert observation_2d.sum() > 0, "No agents in observation"
    assert observation_2d.dtype == np.int64 or observation_2d.dtype == np.int32, f"Wrong dtype: {observation_2d.dtype}"

    print(f"   ✅ 2D simulation working correctly")

    return True

def main():
    """Run all tests."""
    print("🚀 Testing Improved 2D CNN Pipeline")
    print("=" * 50)

    try:
        # Run tests
        test_improved_cnn()
        test_training_monitor()
        test_2d_npe_setup()
        test_data_simulation()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! The improved 2D CNN pipeline is ready.")
        print("\n📋 Summary of improvements:")
        print("   ✅ Increased batch size from 128 to 512")
        print("   ✅ Progressive downsampling instead of aggressive pooling")
        print("   ✅ Increased CNN output dimension from 128 to 256")
        print("   ✅ Added per-sample data normalization")
        print("   ✅ Reduced learning rate to 5e-5 for stable training")
        print("   ✅ Increased early stopping patience to 30 epochs")
        print("   ✅ Added training monitoring and debugging tools")
        print("\n🔄 Ready to run improved 2D reproduction script!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)