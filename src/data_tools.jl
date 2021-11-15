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

"""
    load_precomputed_data()

Loads the precomputed data that is saved in the package. It is computed from ERA5 T21 u/v data. Returns in order

* `S`, `qg3ppars`, `ψ_0`, `q_0`
* Forcing, Parameters, Streamfunction initial conditions, vorticity initial conditions
"""
function load_precomputed_data()

    path = joinpath(dirname(@__FILE__), "..", "data/")

    @load string(path,"t21-precomputed-S.jld2") S
    @load string(path,"t21-precomputed-p.jld2") qg3ppars
    @load string(path,"t21-precomputed-sf.jld2") ψ_0
    @load string(path,"t21-precomputed-q.jld2") q_0

    return S, qg3ppars, ψ_0, q_0
end
