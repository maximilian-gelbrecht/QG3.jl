lat_to_colat(lat) = π/2 .- lat

"""
    level_index(strings,lvls)

Helper function for indexing netcdf files with many levels.

* string, array with name of levels to be selected
* lvls, array with all levels

"""
function level_index(strings,lvls)
    ind = BitArray(zeros(length(lvls)))
    for ilvl in strings
        ind = ind .| (ilvl .== lvls)
    end
    return ind
end



function seconds(data::T, unit::String) where T<:Number
    if unit ∈ ["h","hours","hour"]
        return data*T(60)*T(60)
    elseif unit ∈ ["d","day","days"]
        return data*T(60)*T(60)*(24)
    else
        error("Not supported unit.")
    end
end


function plot_ticks(p::QG3ModelParameters, name, N_interval,digits=2)
    if name=="lat"
        Ntickrange = p.N_lats
        tickval = p.lats
    elseif name=="lon"
        Ntickrange = p.N_lons
        tickval = p.lons
    else
        error("Not supported name")
    end

    tickrange = 1:N_interval:Ntickrange
    tickvals = []
    for itick in tickrange
        push!(tickvals, round(tickval[itick], digits=digits))
    end
    (tickrange, tickvals)
end
