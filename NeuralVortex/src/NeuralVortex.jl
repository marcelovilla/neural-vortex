module NeuralVortex

# Example public function
greet() = println("Hello World!")

# Bring in internal source files
include("esn.jl")
include("FNO_helper.jl")
include("utils.jl")

# Expose only the main functions for users
export greet,
       build_vor_data_matrix,
       split_data,
       fit_esn,
       grid_search,
       spectral_prediction_to_grid,
       predict_grid,
       compute_error,
       compute_error_fno,
       generate_vort_all_trunc,
       generate_vorticity,
       octa_to_rect_grid,
       gridded_data_fno

end # module NeuralVortex

