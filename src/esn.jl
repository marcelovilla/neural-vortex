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
function build_vor_data_matrix(output_vector)
    n_samples = length(output_vector)
    n_modes = size(output_vector[1][1].vor)[1]
    # Because the spectral coefficients are complex, we separate the real and imaginary parts
    # into different features, having twice as many features as modes.
    n_features = n_modes * 2

    data = zeros(Float64, n_features, n_samples)
    for t in 1:n_samples
        # Get the spectral vorticity coefficients at leapfrog level 2
        vor_spec = output_vector[t][1].vor[:, :, 2]
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
function split_data(data; train_ratio=0.6, val_ratio=0.2)
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
        println("Training with hyperparameters: $hyperparams")

        esn, output_layer = fit_esn(train_data, hyperparams)
        prediction = esn(Generative(predict_len), output_layer)
        loss = sum(abs2, prediction - val_data)

        if loss < best_loss
            best_loss = loss
            best_params = hyperparams
        end
    end

    println("Best hyperparameters: $best_params with loss $best_loss")

    # Train the ESN on the combined training and validation data using the best
    # hyperparameters
    esn, output_layer = fit_esn(hcat(train_data, val_data), best_params)

    return esn, output_layer
end

function spectral_prediction_to_grid(spectral_prediction, reference_alms, spectral_grid)
    n_features = size(spectral_prediction, 1)
    n_modes = Int(n_features / 2)

    prediction_vector = []
    for t in 1:size(spectral_prediction, 2)

        # Real and imaginary parts of the complex coefficients were previously separated
        # into different functions. Now they are combined together again.
        re = spectral_prediction[1:n_modes, t]
        im = spectral_prediction[n_modes+1:end, t]
        values = complex.(re, im)

        # The data needs to be a LowerTriangularArray in order to be transformed back
        # to the grid. The prediction is a simple array but their values can be used
        # to populate a copy of reference data from the original data.
        alms = copy(reference_alms)
        alms .= values

        S = SpectralTransform(spectral_grid)
        grid_prediction = transform(alms, S)
        push!(prediction_vector, grid_prediction)
    end

    return prediction_vector
end


function predict_grid(output_vector, search_space)

    data = build_vor_data_matrix(output_vector)
    train_data, val_data, test_data = split_data(data)
    esn, output_layer = grid_search(train_data, val_data, search_space)

    # Predict all the timesteps in the test set using the generative method
    predict_len = size(test_data, 2)
    spectral_prediction = esn(Generative(predict_len), output_layer)

    # Reference alms is used as reference object to populate with predicted values
    # while keeping the original structure
    reference_alms = output_vector[1][1].vor[:, :, 1][:, 1]

    # Get truncation level from the data. There might be a better way to do this.
    trunc_level = output_vector[1][1].vor.spectrum.mmax - 1

    # Convert spectral prediction to grid space
    spectral_grid = SpectralGrid(trunc=trunc_level, nlayers=1)
    grid_prediction = spectral_prediction_to_grid(spectral_prediction, reference_alms, spectral_grid)

    # Compute the timestep offset where the testing data starts
    offset = size(train_data, 2) + size(val_data, 2)

    return grid_prediction, offset
end


function compute_error(prediction, output_vector, offset, err_func)
    n = length(prediction)
    err = zeros(n)
    for i in 1:n
        ground_truth = output_vector[offset + i][2].grid.vor_grid[:, 1]
        err[i] = err_func(prediction[i], ground_truth)
    end

    return err
end
