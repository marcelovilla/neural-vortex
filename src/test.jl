#### TESTING SIZES OF OUTPUT VECTORS (Trunc_x.jld2 files in run000x folders)

using JLD2

run_folders = ["run_0001", "run_0002", "run_0003", "run_0004", "run_0005"]
truncations = [5, 16, 31, 42, 63]

for (folder, trunc) in zip(run_folders, truncations)
    filepath = joinpath(folder, "trunc_$(trunc).jld2")
    jldopen(filepath, "r") do f
        if haskey(f, "output_vector")
            snaps = length(f["output_vector"])
            println("Trunc=$trunc in $folder → snapshots = $snaps")
        else
            println("Trunc=$trunc in $folder → no output_vector key")
        end
    end
end
