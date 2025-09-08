using Dates
using SpeedyWeather

function generate_vorticity(trunc, period, output_dt; filename="output.jld2")

    spectral_grid = SpectralGrid(trunc=trunc, nlayers=1)
    still_earth = Earth(spectral_grid, rotation=0)
    initial_conditions = RandomVelocity()
    forcing = NoForcing()
    drag = NoDrag()

    output = JLD2Output(; filename=filename, output_dt=output_dt, output_prognostic = true, output_diagnostic = true)
    model = BarotropicModel(spectral_grid; initial_conditions, planet=still_earth, forcing, drag, output=output)
    simulation = initialize!(model)

    run!(simulation, period=period, output=true)
end

function generate_vort_all_trunc()

    truncations = [5, 20]

    for trunc in truncations
        filename = "trunc_$trunc.jld2"
        println("Running truncation $trunc → saving to $filename")
        generate_vorticity(trunc, Day(20), Hour(1); filename=filename)
    end
end
