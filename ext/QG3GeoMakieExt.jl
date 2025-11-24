module QG3GeoMakieExt

using QG3
using GeoMakie

function QG3.animate(qg3ppars::QG3ModelParameters, solu::AbstractArray;
                    output_filename = "data_animation.mp4", 
                    fixed_level_idx = 1,
                    dest = "+proj=moll",
                    variable_name = "Potential Vorticity",
                    framerate = 5 
                    )
    
    lons = range(-180,180,qg3ppars.N_lons)    
    lats = rad2deg.(qg3ppars.lats)    
    levs = [800, 500, 200] 
    times = range(1,size(solu,4))

    begin
        my_theme = merge(theme_latexfonts(), theme_minimal())
        set_theme!(my_theme, fontsize = 24, font = "Helvetica", color = :black)

        fig = Figure(size = (1100, 700))

        fixed_level_hpa = levs[fixed_level_idx]

        fig[0, 1:2] = Label(fig, "Data Animation (lon-lat map) at $(fixed_level_hpa) hPa", fontsize=24, padding=(0, 10, 20, 10))

        ax = GeoAxis(fig[1, 1],
                    dest = dest, 
                    title = "Time Step: 1" 
                    )
        
        clims = (minimum(solu), maximum(solu))

        data_slice = Observable(solu[:, :, fixed_level_idx, 1])
        
        sf = GeoMakie.surface!(ax, lons, lats, data_slice,
                    colormap = :balance, 
                    colorrange = clims
                    )
        
        lines!(ax, GeoMakie.coastlines(ax), color=:black, overdraw=true)
        
        Colorbar(fig[1, 2], sf, label="$(variable_name) Value")

        n_frames = length(times)
        
        GeoMakie.record(fig, output_filename, 1:n_frames, framerate = framerate) do frame_index
           ax.title = "Time Step: $(frame_index)"
            data_slice[] = solu[:, :, fixed_level_idx, frame_index]
        end

        println("Animation complete. File saved to $(output_filename).")  
    end
    
end

end
