# 2D NPE Slurm Scripts

This directory contains Slurm batch scripts for running the 2D NPE functionality on compute clusters.

## 🚀 **New 2D Scripts**

### `run_main_2d.sh` (Standard 2D Run)
- **Purpose**: Standard production run with 2D spatial CNN processing
- **Resources**: 32GB RAM, 8 CPUs, 12 hours
- **Configuration**: 100×50 lattice, 8 SNPE rounds, 5000 samples/round
- **Use case**: Regular 2D NPE experiments

### `run_main_2d_light.sh` (Quick Testing)
- **Purpose**: Lightweight version for testing 2D functionality
- **Resources**: 16GB RAM, 4 CPUs, 4 hours
- **Configuration**: 50×25 lattice, 5 SNPE rounds, 2000 samples/round
- **Use case**: Rapid prototyping, debugging, initial validation

### `run_main_2d_heavy.sh` (High-Resolution Production)
- **Purpose**: Intensive run for high-quality results with full spatial resolution
- **Resources**: 64GB RAM, 16 CPUs, 24 hours
- **Configuration**: 200×100 lattice, 12 SNPE rounds, 8000 samples/round
- **Use case**: Publication-quality results, detailed spatial analysis

### `run_comparison_1d_vs_2d.sh` (Comparative Study)
- **Purpose**: Direct comparison between 1D (column counts) and 2D (spatial CNN) approaches
- **Resources**: 32GB RAM, 8 CPUs, 16 hours
- **Configuration**: Runs both versions with identical parameters
- **Use case**: Method validation, performance comparison

## 📊 **Key Differences from 1D Scripts**

### Enhanced Resource Requirements
- **Memory**: Increased RAM allocation (2-4x) due to CNN processing and 2D data storage
- **CPUs**: More cores for parallel CNN operations and data processing
- **Time**: Longer runtimes due to increased computational complexity

### 2D-Specific Parameters
- **`--use_2d_data`**: Enables 2D spatial processing with CNN architecture
- **Batch Size**: Reduced for 2D data (64 vs 128) due to memory constraints
- **Learning Rate**: Often reduced for CNN training stability
- **Hidden Features**: CNN embedding dimensions for spatial processing

### Output Differences
- **Additional Visualizations**: 2D spatial plots alongside traditional 1D plots
- **Enhanced Data**: Full spatial information preserved for analysis
- **CNN Architecture**: Convolutional layers for spatial pattern recognition

## 🔧 **Usage Instructions**

### Submit a job:
```bash
sbatch slurm/run_main_2d.sh              # Standard 2D run
sbatch slurm/run_main_2d_light.sh        # Quick test
sbatch slurm/run_main_2d_heavy.sh        # High-resolution run
sbatch slurm/run_comparison_1d_vs_2d.sh  # Comparison study
```

### Monitor progress:
```bash
squeue -u $USER                          # Check job status
tail -f slurm/outputs/snpe_*_<job_id>.txt # Watch output
```

### Check results:
```bash
ls results/workflow_*                     # List result directories
ls results/comparison_*                   # List comparison results
```

## 📈 **Expected Performance**

### 2D Advantages:
- **Richer Information**: Full spatial structure preserved
- **Better Inference**: CNN can learn spatial patterns and correlations
- **Enhanced Analysis**: Spatial uncertainty quantification
- **More Realistic**: Matches natural 2D biological processes

### Computational Trade-offs:
- **Memory**: 2-4x higher RAM usage for 2D data and CNN processing
- **Time**: 1.5-3x longer training due to CNN complexity
- **GPU**: More intensive GPU utilization for convolutions

## 🧪 **Recommended Workflow**

1. **Start with Light**: Use `run_main_2d_light.sh` to validate setup
2. **Standard Production**: Use `run_main_2d.sh` for regular experiments
3. **Comparison Study**: Use `run_comparison_1d_vs_2d.sh` to quantify benefits
4. **High-Resolution**: Use `run_main_2d_heavy.sh` for final publication results

## 💡 **Parameter Tuning Tips**

### For CNN Training:
- Reduce batch size if memory issues occur
- Lower learning rate for training stability
- Increase `stop_after_epochs` for convergence
- Monitor GPU memory usage

### For Large Lattices:
- Increase memory allocation proportionally
- Consider gradient accumulation for large batches
- Use mixed precision training if available
- Monitor convergence carefully

## 🔍 **Troubleshooting**

### Common Issues:
- **Out of Memory**: Reduce batch size or lattice dimensions
- **Slow Convergence**: Adjust learning rate or network architecture
- **Job Timeout**: Increase time allocation or reduce problem size
- **CNN Errors**: Check PyTorch/CUDA compatibility

### Performance Optimization:
- Use GPU-optimized conda environment
- Enable CUDA memory caching
- Monitor resource utilization
- Profile bottlenecks with `nvidia-smi`