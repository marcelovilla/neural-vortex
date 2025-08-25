# FNO - Helper script"
  
# From src/utils.jl and src/helper.ipynb, the output_vector from SpeedyWeather 
# needs to be converted to gridded data, s.t. the 2D FNO can work with it"

using JLD2
using SpeedyWeather.RingGrids

include("../../src/utils.jl")
include("../../src/helper.jl")

# a) from octrahedral to rectangular grid (as a resolution, choosing 192 x 96 since 
# that corresponds to our highest truncation level T = 63)
# using RingGrids.jl library

 # function octa_to_rect_grid(field; nlat=96, nlon = 192) {

    # Build target grid and target geometry
  #  build_grid = RingGrids.FullGaussianGrid(nlat, nlon)
 #   target_geom = RingGrids.GridGeometry(build_grid)

    # Set up interpolator
   # interpolator = RingGrids.AnvilInterpolator(field.grid, nlat)

    # Allocate output
    # out = zeros(Float32, nlat, nlon)

    # Build a target field living on the interpolator’s grid
  #  target_grid = RingGrids.FullGaussianGrid(nlat)
 #   target_field = RingGrids.Field(Float32, target_grid)
#
    # Interpolate!
#    RingGrids.interpolate!(target_field, field, interpolator)

 #   return reshape(Array(target_field.values, nlat, nlon))
# end
# }

# function octa_to_rect_grid(field; nlat=48, nlon = 192)
    # Build the target grid
    #target_grid = RingGrids.FullGaussianGrid(nlat)

    # Interpolate directly
   # out_field = RingGrids.interpolate(target_grid, field)

    # Reshape values to (nlat, nlon)
   # arr = parent(out_field)
   # return reshape(arr, 2*nlat, nlon)
# end

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

# b) from spectral to gridded data

function gridded_data_fno(filename; nlat=96, nlon=192)
    data = JLD2.load(filename)
    nt = length(data["output_vector"])

    snapshots = [
        begin
            diag = data["output_vector"][t][2]  # DiagnosticVariables([2]) at time t
            u   = octa_to_rect_grid(diag.grid.u_grid;   nlat=nlat, nlon=nlon)
            v   = octa_to_rect_grid(diag.grid.v_grid;   nlat=nlat, nlon=nlon)
            vor = octa_to_rect_grid(diag.grid.vor_grid; nlat=nlat, nlon=nlon)
            cat(u, v, vor; dims=3)   # Stack the channel variables u,v,vor
        end
        for t in 1:nt
    ]
    X = stack(snapshots)                  # (nlat, nlon, 3, time)
    # X = permutedims(X, (3, 1, 2, 4))       # (channels=3, nlat, nlon, time)
    println(size(X))
    train_size = 384 # 80% of total sample size 481, for training (rest for validation)
    x_train = X[:, :, :,1:(train_size)-1] # x
    y_train = X[:, :, :,2:(train_size)] # f(x) = y = x + dx
    x_valid = X[:, :, :,train_size:end-1]
    y_valid = X[:, :, :,(train_size)+1:end]
    return x_train, y_train, x_valid, y_valid
end
