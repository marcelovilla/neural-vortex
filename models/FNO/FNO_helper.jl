# FNO - Helper script"
  
# Using src/utils.jl and src/helper.ipynb to generate output_vector from SpeedyWeather 
# Vector needs to be converted from octrahedral gaussian grid to 
# gridded data, s.t. the 2D FNO can work with it"

using JLD2
using SpeedyWeather.RingGrids

include("../../src/utils.jl")
include("../../src/helper.jl")

# a) From octrahedral gaussian to rectangular grid, choosing resolution 192 x 96  

using SpeedyWeather.RingGrids

"""
Interpolate an OctahedralGaussianField to a FullGaussianGrid and return
the values as a `(nlat, nlon)` array.
"""
function octa_to_rect_grid(field; nlat=96, nlon = 192)
    # Build target grid: FullGaussianGrid requires "half-lats"
    target_grid = FullGaussianGrid(nlat ÷ 2)

    # Interpolate directly
    out_field = interpolate(target_grid, field)

    # Extract the array of values
    arr = parent(out_field)

    # Derive shape from the grid itself
    latitudes  = get_latd(target_grid)   # vector of length nlat
    longitudes = get_lond(target_grid)   # vector of length nlon

    nlat_actual = length(latitudes)
    nlon_actual = length(longitudes)

    # println("Target grid shape: $(nlat_actual) × $(nlon_actual)")
    # println("Interpolated array length: $(length(arr))")

    # Reshape consistently with grid geometry
    return reshape(arr, nlat_actual, nlon_actual)
end

# b) Collecting gridded data only (output vector still contains spectral and gridded data)

function gridded_data_fno(filename; nlat=96, nlon=192, period=Day(20), output_dt=Hour(1))
    data = JLD2.load(filename)
    nt = length(data["output_vector"])

    snapshots = [
        begin
            diag = data["output_vector"][t][2]  # DiagnosticVariables([2]) at time t
            u   = octa_to_rect_grid(diag.grid.u_grid;   nlat=nlat, nlon=nlon)
            v   = octa_to_rect_grid(diag.grid.v_grid;   nlat=nlat, nlon=nlon)
            vor = octa_to_rect_grid(diag.grid.vor_grid; nlat=nlat, nlon=nlon)
            cat(u, v, vor; dims=3)   # Stack channel variables u,v,vor
        end
        for t in 1:nt
    ]
    X = stack(snapshots) # (nlat, nlon, 3, time)


    """
    In src: Internal adjustments of time stepping dt (higher trunc -) smaller dt -) uneven sample size)
    post fixing this by only reading values every 1 Hour (leads to even sample size)
    """
    ##### FIX: crop to exactly period / output_dt samples #####
    n_expected = Int(round(period / output_dt)) + 1  # +1 if you want t=0 included
    X = X[:, :, :, 1:min(n_expected, size(X,4))]

    println("Final shape = ", size(X))
    return X

    function split_train_valid(X; train_ratio=0.8) # using 80% of data for training (481*0.8), 20% for validation
        n_samples = size(X, 4)  # time dimension (last axis)
        train_size = Int(floor(train_ratio * n_samples))
    
        x_train = X[:, :, :, 1:(train_size-1)] # x
        y_train = X[:, :, :, 2:(train_size)] # f(x) = y = x + dx
        x_valid = X[:, :, :, (train_size):end-1]
        y_valid = X[:, :, :, (train_size+1):end]
    
        return x_train, y_train, x_valid, y_valid
    end

end
