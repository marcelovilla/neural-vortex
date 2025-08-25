using Dates

include("utils.jl")

function generate_vort_all_trunc()

    truncations = [5, 16, 31, 42, 63]

    for trunc in truncations
        filename = "trunc_$trunc.jld2"
        println("Running truncation $trunc → saving to $filename")
        generate_vorticity(trunc, Day(20), Hour(1); filename=filename)
    end
end
