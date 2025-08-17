using SpeedyWeather

function generate_vorticity(trunc, period, output_dt)

    spectral_grid = SpectralGrid(trunc=trunc, nlayers=1)
    still_earth = Earth(spectral_grid, rotation=0)
    initial_conditions = RandomVelocity()
    forcing = NoForcing()
    drag = NoDrag()

    output = JLD2Output(output_dt=output_dt)
    model = BarotropicModel(spectral_grid; initial_conditions, planet=still_earth, forcing, drag, output=output)
    simulation = initialize!(model)

    run!(simulation, period=period, output=true)
end
