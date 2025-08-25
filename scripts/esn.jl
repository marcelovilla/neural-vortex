root_dir = dirname(@__DIR__)
include(joinpath(root_dir, "src/esn.jl"))

data = build_vor_data_matrix(joinpath(root_dir, "run_0001/output.jld2"))
train_data, val_data, test_data = split_data(data)
search_space = (
    reservoir_size = [512, 1024],
    spectral_radius = [0.9, 0.95],
    sparsity = [0.03, 0.05],
    input_scaling = [0.15, 0.3],
    ridge_param = [1e-5],
)
esn, output_layer = grid_search(train_data, val_data, search_space)
