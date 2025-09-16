# NPE Demo Notebooks: 1D vs 2D Comparison

This directory contains marimo notebooks demonstrating both traditional and enhanced NPE approaches.

## 📊 **Available Notebooks**

### `demo.py` - Traditional 1D NPE
- **Data Format**: Column counts (compressed spatial data)
- **Neural Network**: Standard dense layers
- **Focus**: Classic NPE workflow and validation
- **Computational**: Lower memory/time requirements
- **Use Case**: Standard parameter inference tasks

### `demo_2d.py` - Enhanced 2D NPE 🆕
- **Data Format**: Full 2D spatial grids (complete spatial information)
- **Neural Network**: CNN + dense layers for spatial processing
- **Focus**: Spatial pattern learning and enhanced inference
- **Computational**: Higher memory/time for CNN processing
- **Use Case**: Spatially-aware parameter inference

## 🚀 **Running the Notebooks**

### Prerequisites:
```bash
# Install marimo if not already available
pip install marimo

# Ensure you're in the project root directory
cd /path/to/NPE_for_RW_Inference
```

### Launch notebooks:
```bash
# Traditional 1D approach
marimo run notebooks/demo.py

# Enhanced 2D approach (NEW!)
marimo run notebooks/demo_2d.py

# Or edit mode for interactive development
marimo edit notebooks/demo_2d.py
```

## 🔍 **Key Differences Demonstrated**

### Data Representation:
- **1D Demo**: Shows traditional column count compression
- **2D Demo**: Reveals full spatial structure preservation

### Visualization:
- **1D Demo**: Standard column plots and corner plots
- **2D Demo**: Spatial heatmaps, evolution animations, CNN architecture

### Information Content:
- **1D Demo**: Limited to column-wise aggregation
- **2D Demo**: Complete spatial patterns and correlations

### Scientific Insight:
- **1D Demo**: Parameter inference with spatial data loss
- **2D Demo**: Spatially-aware inference with pattern recognition

## 📈 **Educational Progression**

### Recommended Learning Path:
1. **Start with `demo.py`**: Understand basic NPE concepts
2. **Explore `demo_2d.py`**: See enhanced 2D capabilities
3. **Compare Results**: Understand information preservation benefits
4. **Run Experiments**: Test both approaches on your data

### Key Learning Outcomes:
- **Data Format Impact**: How spatial compression affects inference
- **CNN Benefits**: Spatial feature learning for parameter estimation
- **Trade-offs**: Computational cost vs. information content
- **Applications**: When to use 1D vs 2D approaches

## 🎯 **Practical Applications**

### Use 1D Approach When:
- Computational resources are limited
- Spatial patterns are not crucial for parameter inference
- Quick prototyping or baseline results needed
- Column-wise aggregation captures sufficient information

### Use 2D Approach When:
- Spatial patterns are important for parameter estimation
- Maximum inference accuracy is required
- Scientific insight into spatial processes is desired
- Computational resources support CNN processing

## 🔧 **Technical Features Demonstrated**

### `demo_2d.py` Unique Features:
- **Side-by-side Comparison**: 1D vs 2D data visualization
- **Spatial Evolution**: Before/after spatial pattern analysis
- **CNN Architecture**: Network design for spatial processing
- **Information Analysis**: Quantitative comparison of data content
- **Integration Guide**: How 2D fits into existing workflows

### Interactive Elements:
- **Real-time Visualization**: See spatial patterns emerge
- **Parameter Exploration**: Adjust U, P, T and observe effects
- **Architecture Inspection**: Understand CNN feature extraction
- **Performance Metrics**: Compare 1D vs 2D quantitatively

## 💡 **Tips for Exploration**

### Customization:
```python
# In demo_2d.py, try different lattice sizes
Lx = 50   # Smaller for faster processing
Ly = 25
# or
Lx = 120  # Larger for more detail
Ly = 60

# Experiment with parameters
U = 0.2   # Lower occupancy
P = 0.9   # Higher movement probability
T = 200   # Longer simulation
```

### Comparative Analysis:
- Run both notebooks with identical parameters
- Compare information content metrics
- Analyze spatial correlation preservation
- Evaluate visualization differences

### Performance Profiling:
- Monitor memory usage during CNN processing
- Compare simulation times: 1D vs 2D
- Profile CNN forward pass efficiency
- Analyze spatial feature learning

## 📚 **Further Reading**

- **Main Documentation**: `../README.md` - Project overview
- **Slurm Scripts**: `../slurm/README_2D.md` - Cluster deployment
- **Implementation**: `../src/` - Source code with 2D extensions
- **Results**: `../results/` - Example outputs and analysis

## 🎉 **Getting Started**

**Quick Start for 2D Demo:**
```bash
cd NPE_for_RW_Inference
marimo run notebooks/demo_2d.py
```

Navigate through the cells to see:
1. 2D simulator capabilities
2. Spatial data preservation
3. CNN architecture design
4. Information content analysis
5. Integration with NPE workflow

The 2D approach represents a significant enhancement to spatial parameter inference - enjoy exploring the new capabilities! 🚀