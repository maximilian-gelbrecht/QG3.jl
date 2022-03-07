lat_to_colat(lat::T) where {T} = T(π/2) - lat


"""
    make3d(A::AbstractArray{T,2})

    repeats an array three times to turn it into (3 x size(A,1) x size(A,2)), for the fully matrix version of model.
"""
function make3d(A::AbstractArray{T,2}) where T<:Number
    togpu(reshape(A,1,size(A,1),size(A,2)))
end


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

# version just for post processing of solutions in 4D (not suitable for the model
function qprimetoψ(p::QG3Model{T}, q::AbstractArray{T,4}) where T<:Number

    out = similar(q)

    for it ∈ 1:size(out,4)
        out[:,:,:,it] = qprimetoψ(p, q[:,:,:,it])
    end

    return out
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

"""
    zeros_SH(p::QG3Model{T})

Returns a zero array in the dimensions of the SH. 
"""
function zeros_SH(p::QG3Model{T}) where {T}
    if isongpu(p)
        return reorder_SH_gpu(zeros_SH(p.p), p.p)
    else 
        return zeros_SH(p.p)
    end 
end
zeros_SH(p::QG3ModelParameters{T}) where {T} = zeros(T, 3, p.L, p.M)

"""
    zeros_Grid(p::QG3Model{T})

Returns a zero array in the dimensions of the grid. 
"""
function zeros_Grid(p::QG3Model{T}) where {T}
    if isongpu(p)
        return CUDA.CuArray(zeros_Grid(p.p))
    else 
        return zeros_Grid(p.p)
    end 
end
zeros_Grid(p::QG3ModelParameters{T}) where {T} = zeros(T, 3, p.N_lats, p.N_lons)





