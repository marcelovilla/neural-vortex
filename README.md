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
