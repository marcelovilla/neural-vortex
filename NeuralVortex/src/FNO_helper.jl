# FNO - Helper script"
  
# Using src/utils.jl and src/helper.ipynb to generate output_vector from SpeedyWeather 
# Vector needs to be converted from octrahedral gaussian grid to 
# gridded data, s.t. the 2D FNO can work with it"

using JLD2
using SpeedyWeather.RingGrids
using CairoMakie
CM = CairoMakie

# a) From octrahedral gaussian to rectangular grid, choosing resolution 192 x 96  

using SpeedyWeather.RingGrids

"""
Interpolate an OctahedralGaussianField to a FullGaussianGrid and return
the values as a `(nlat, nlon)` array.
"""
function octa_to_rect_grid(field; nlat=96, nlon=192)
    target_grid = FullGaussianGrid(nlat ÷ 2)
    out_field   = interpolate(target_grid, field)  # Field on FullGaussianGrid
    arr = Array(out_field)  # already shaped (nlat, nlon)
    return arr   # return both
end

# b) Collecting gridded data only (output vector still contains spectral and gridded data)

function gridded_data_fno(filename; nlat=96, nlon=192)
    data = JLD2.load(filename)
    nt = length(data["output_vector"])
    diag1 = data["output_vector"][1][2]
    println("Raw octahedral vorticity: ", extrema(diag1.grid.vor_grid[:,1]))
    vor_arr = octa_to_rect_grid(diag1.grid.vor_grid; nlat=nlat, nlon=nlon)
    println("After octa_to_rect_grid → FullGaussian: ", extrema(vor_arr))

    snapshots = [
        begin
            #everything in green can be undone to feature u, v and vor, not just vor
            diag = data["output_vector"][t][2]  # DiagnosticVariables([2]) at time t
            # u   = octa_to_rect_grid(diag.grid.u_grid;   nlat=nlat, nlon=nlon)
            # v   = octa_to_rect_grid(diag.grid.v_grid;   nlat=nlat, nlon=nlon)
            vor = octa_to_rect_grid(diag.grid.vor_grid; nlat=nlat, nlon=nlon)
            # cat(u, v, vor; dims=3)   # Stack the channel variables u,v,vor
            reshape(vor, nlat, nlon, 1) 
        end
        for t in 1:nt
    ]
    X = cat(snapshots...; dims=4)                  # (nlat, nlon, 3, time)
    println(size(X))
    println("Max of concatenated X:", extrema(X[:,:,1,:]))
    train_size = 384 # 80% of total sample size 481, for training (rest for testing)
    x_train = X[:, :, :,1:(train_size)-1] # x
    y_train = X[:, :, :,2:(train_size)] # f(x) = y = x + dx
    x_test = X[:, :, :,train_size:end-1]
    y_test = X[:, :, :,(train_size)+1:end]
    println("x_train extrema just before save: ", extrema(x_train))
    println("x_test extrema just before save: ", extrema(x_test))
    return x_train, y_train, x_test, y_test
end

function rmse(pred, x_test)
    return sqrt(mean((pred .- x_test).^2))
end 

function compute_error_fno(pred, x_test)
    n = size(x_test, 4)
    err = zeros(n)
    for i in 1:n
        err[i] = rmse(pred[:,:,1,i], x_test[:,:,1,i])
    end 
    return err
end
