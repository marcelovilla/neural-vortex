using CairoMakie
using SpeedyWeather.LowerTriangularArrays
using NeuralVortex


search_space = (
    reservoir_size = [512, 1024],
    spectral_radius = [0.9, 0.95],
    sparsity = [0.03, 0.05],
    input_scaling = [0.15, 0.3],
    ridge_param = [1e-5],
)

# Initialize an empty dict to store the RMSE series for both truncation levels
errors = Dict()

for (trunc_level, file_path) in Dict("T5" => "run_0001/trunc5.jld2", "T20" => "run_0002/trunc20.jld2")

    # Load results from SpeedyWeather simulation
    output_vector = JLD2.load(joinpath(root_dir, file_path))["output_vector"]

    # Predict over the test timesteps in grid space
    grid_prediction, offset = predict_grid(output_vector, search_space)

    for i in [1, 7, length(grid_prediction)]
        t = i + offset;
        ground_truth = output_vector[t][2].grid.vor_grid[:, 1]
        prediction = grid_prediction[i]

        fig = heatmap(ground_truth, title="Ground truth t=$t")
        save(joinpath(root_dir, "figures/ESN_ground_truth_$(trunc_level)_t$t.png"), fig)

        fig = heatmap(prediction, title="Grid prediction t=$t")
        save(joinpath(root_dir, "figures/ESN_prediction_$(trunc_level)_t$t.png"), fig)
    end

    errors[trunc_level] = compute_error(grid_prediction, output_vector, offset, rmse)
end

fig = Figure(resolution = (800, 400), fontsize = 13)
ax  = Axis(fig[1, 1];
    title = "RMSE — T5 vs T20",
    xlabel = "Forecast step",
    ylabel = "RMSE",
    # yscale = log10,
    xgridvisible = true,
    ygridvisible = true)
lines!(ax, errors["T5"];  linewidth = 3, label = "T5")
lines!(ax, errors["T20"]; linewidth = 3, label = "T20")
axislegend(ax; position = :rb, framevisible = true)

save(joinpath(root_dir, "figures", "ESN_rmse_T5_T20.png"), fig; px_per_unit = 2)
