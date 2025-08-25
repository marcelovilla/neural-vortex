using Base.Iterators: product
using JLD2
using ReservoirComputing
using SpeedyWeather

# Use the @kwdef macro to define a struct that can be constructed using keywords instead
# of relying on positional arguments.
@kwdef struct Hyperparams
    reservoir_size::Int
    spectral_radius::Float64
    sparsity::Float64
    input_scaling::Float64
    ridge_param::Float64
end

"""
Build a (n_features x n_samples) matrix of spectral vorticity coefficients from a
SpeedyWeather JLD2 output.

Each column corresponds to one output time and each row is a spherical-harmonic (ℓ,m) mode
from `PrognosticVariables.vor` at leapfrog level 1 (`vor[:, :, 1]`).

# Arguments
- `file_path`: Path to the `.jld2` file produced by `JLD2Output`.

# Returns
- `Matrix{Float64}`: Spectral vorticity coefficients with shape `(n_modes * 2, n_times)`.
"""
function build_vor_data_matrix(file_path)
    output = JLD2.load(file_path)["output_vector"]

    n_samples = length(output)
    n_modes = size(output[1][1].vor)[1]
    # Because the spectral coefficients are complex, we separate the real and imaginary parts
    # into different features, having twice as many features as modes.
    n_features = n_modes * 2

    data = zeros(Float64, n_features, n_samples)
    for t in 1:n_samples
        vor_spec = output[t][1].vor[:, :, 1]
        # Separate real and imaginary parts into different features
        data[:, t] = vcat(real(vor_spec), imag(vor_spec))
    end

    return data
end

"""
Split a time-ordered feature matrix into contiguous train/validation/test blocks.

The input `data` is assumed to be shaped (n_features x n_samples).

# Arguments
- `data`: Features by samples matrix (with columns = timesteps).
- `train_ratio=0.7`: Fraction of samples for the training block.
- `val_ratio=0.15`: Fraction for the validation block.

The test fraction is the remainder `1 - (train_ratio + val_ratio)`.

# Returns
- `(train_data, val_data, test_data)`: Three matrices with the same number of rows as
  `data` and column counts determined by the ratios.
"""
function split_data(data; train_ratio=0.7, val_ratio=0.15)
    train_ratio + val_ratio <= 1 || error("Ratios sum should not exceed 1")

    n_samples = size(data, 2)
    n_train = Int(floor(train_ratio * n_samples))
    n_val = Int(floor(val_ratio * n_samples))

    train_data = data[:, 1:n_train]
    val_data = data[:, n_train+1:n_train+n_val]
    test_data = data[:, n_train+n_val+1:end]

    return train_data, val_data, test_data
end


function fit_esn(train_data, hyperparams)

    # The target signal corresponds to the input signal shifted by one timestep into
    # the future
    u_train = train_data[:, 1:end-1]
    y_train = train_data[:, 2:end]

    esn = ESN(
        u_train,
        size(u_train, 1),
        hyperparams.reservoir_size;
        reservoir=rand_sparse(
            radius=hyperparams.spectral_radius,
            sparsity=hyperparams.sparsity
        ),
        input_layer=scaled_rand(scaling=hyperparams.input_scaling),
    )

    training_method = StandardRidge(hyperparams.ridge_param)
    output_layer = train(esn, y_train, training_method)

    return esn, output_layer
end

function grid_search(train_data, val_data, search_space)
    best_loss = Inf
    best_params = nothing

    # Number of timesteps to predict during validation
    predict_len = size(val_data, 2)

    param_names = keys(search_space);
    for param_values in product(values(search_space)...)
        hyperparams = Hyperparams(; NamedTuple{param_names}(param_values)...)
        println(hyperparams)
        esn, output_layer = fit_esn(train_data, hyperparams)
        prediction = esn(Generative(predict_len), output_layer)
        loss = sum(abs2, prediction - val_data)

        if loss < best_loss
            best_loss = loss
            best_params = hyperparams
        end
    end

    # Train the ESN on the combined training and validation data using the best
    # hyperparameters
    esn, output_layer = fit_esn(hcat(train_data, val_data), best_params)

    return esn, output_layer
end
