# 📁 Repository Structure

The repository contains the important folders:
```
|-- configs/                # Configuration files
|-- docs/                   # Documentation
|-- evaluation/             # Scripts to reproduce results and datasets initialization
|-- notebooks/              # Jupyter notebooks for inference (of any VFM at any scale) and attention map visualizations
|-- src/                    # Source code of the project
|-- test/                   # Unit tests to compare model efficiency
```


# 🔍 Tests

We provide unit tests to evaluate the efficiency of different model configurations, including forward/backward runtime, GPU memory usage, GFLOPs, and number of parameters.  
All tests are available in the `test/` directory and we provide the test_results.json file with pre-computed results for reference computed on a 1 A100 40GB GPU.

