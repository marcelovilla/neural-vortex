root_dir = dirname(@__DIR__)
include(joinpath(root_dir, "src/utils.jl"))

generate_vort_all_trunc()
