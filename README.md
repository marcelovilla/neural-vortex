# neural-vortex
Neural Vortex: A Julia implementation of Echo State Networks (ESNs) and Fourier Neural Operators (FNOs) for vorticity prediction in dynamical systems 

## How to run

1. Install the required Julia packages by running the following command in the Julia REPL:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

2. Run the vorticity generation script:
```bash
julia scripts/generating_data.jl
```

3. Run the ESN predictions on the generated data:
```bash
julia scripts/esn_predictions.jl
```
### For the FNO:
if you already are on a GPU, you can run the ``` FNO/FNO2D.ipynb ```, which includes data generation
If you use an external GPU through a different UI (which does not take ipynbs), use ``` FNO/FNO2D_gpu.jl ```
Change all filepaths accordingly!

This project includes a package called "NeuralVortex", which can be found under NeuralVortex/ 
