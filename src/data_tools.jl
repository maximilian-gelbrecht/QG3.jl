lat_to_colat(lat::T) where {T} = T(π/2) - lat

"""
    hours(x, m::QG3Model)

Convert from model time units to hours
"""
hours(x, m::QG3Model) = hours(x, m.p)
hours(x, p::QG3ModelParameters) = x * p.time_unit * 24

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

"""
# version just for post processing of solutions in 4D (not suitable for the model
function qprimetoψ(p::QG3Model{T}, q::AbstractArray{T,4}) where T<:Number

    out = similar(q)

    for it ∈ 1:size(out,4)
        out[:,:,:,it] = qprimetoψ(p, q[:,:,:,it])
    end

    return out
end
"""

"""
    load_precomputed_data(GPU=false)

Loads the precomputed data that is saved in the package. It is computed from ERA5 T21 u/v data. Returns in order

* `S`, `qg3ppars`, `ψ_0`, `q_0`
* Forcing, Parameters, Streamfunction initial conditions, vorticity initial conditions

If `GPU==true` returns those in GPU SPH order as `CuArrays`, otherwise uses CPU SPH order. 
"""
function load_precomputed_data(; GPU=false)

    path = joinpath(dirname(@__FILE__), "..", "data/")

    @load string(path,"t21-precomputed-S.jld2") S
    @load string(path,"t21-precomputed-p.jld2") qg3ppars
    @load string(path,"t21-precomputed-sf.jld2") ψ_0
    @load string(path,"t21-precomputed-q.jld2") q_0

    if GPU 
        S, qg3ppars, ψ_0, q_0 = QG3.reorder_SH_gpu(S, qg3ppars), togpu(qg3ppars), QG3.reorder_SH_gpu(ψ_0, qg3ppars), QG3.reorder_SH_gpu(q_0, qg3ppars)
    end 

    return S, qg3ppars, ψ_0, q_0
end

"""
    zeros_SH(p::QG3Model{T})

Returns a zero array in the dimensions of the SH. 
"""
function zeros_SH(p::QG3Model{T}; kwargs...) where T
    if isongpu(p)
        return reorder_SH_gpu(zeros_SH(p.p; kwargs...), p.p)
    else 
        return zeros_SH(p.p; kwargs...)
    end 
end

function zeros_SH(p::QG3ModelParameters{T}; N_levels::Int=3, N_batch::Int=0) where T   
    if N_batch > 0 
        return zeros(T, N_levels, p.L, p.M, N_batch)
    else 
        return zeros(T, N_levels, p.L, p.M)
    end 
end 

"""
    zeros_Grid(p::QG3Model{T})

Returns a zero array in the dimensions of the grid. 
"""
function zeros_Grid(p::QG3Model{T}; kwargs...) where T
    if isongpu(p)
        return CUDA.CuArray(zeros_Grid(p.p; kwargs...))
    else 
        return zeros_Grid(p.p; kwargs...)
    end 
end
function zeros_Grid(p::QG3ModelParameters{T}; N_levels::Int=3, N_batch::Int=0) where T
    if N_batch > 0 
        return zeros(T, N_levels, p.N_lats, p.N_lons, N_batch)
    else 
        return zeros(T, N_levels, p.N_lats, p.N_lons)
    end 
end 

"""
    kinetic_energy(ψ::AbstractArray{T,3}, m::QG3Model{T}) where T 

Computes the kinetic energy per layer
"""
function kinetic_energy(ψ::AbstractArray{T,3}, m::QG3Model{T}) where T 
    E_kin = transform_SH((u(ψ, m).^2 .+ v(ψ, m).^2) ./ 2, m)

    return E_kin[:,1,1]
end 

function kinetic_energy(ψ::AbstractArray{T,4}, m::QG3Model{T}) where T 
    E_kin = transform_SH((u(ψ, m).^2 .+ v(ψ, m).^2) ./ 2, m)

    return E_kin[:,1,1,:]
end 

"""
    KineticEnergyCallback{T}(m::QG3Model)

Initializes a callback to compute the kinetic energy. Can be called with `(u, t, integrator)`
and therefore used with a `SavingCallback`.

To initialize a `SavingCallback` you can e.g. do
```julia
vals = SavedValues(eltype(m), Vector{eltype(m)})
sol = solve(prob; cb=SavingCallback(KineticEnergyCallback(m), vals))
vals.saveval 
```
"""
struct KineticEnergyCallback{T}
    m::QG3Model{T}
end 

function (cb::KineticEnergyCallback)(u, t, integrator)
    kinetic_energy(qprimetoψ(cb.m, u), cb.m)
end 

"""
    Generate Gaussian grid with proper Gaussian latitudes using Legendre polynomials.
    Returns latitudes in radians, sorted from North to South (decreasing order).
"""
function gaussian_grid(n_lat::Int; n_lon::Int=2*n_lat, iters::Int=100, tol::Real=1e-10)
    
    # Longitudes (equally spaced)
    lon_step = 2π / n_lon
    longitudes = range(0, 2π - lon_step, length=n_lon)
    
    # Gaussian latitudes (using Legendre polynomial roots)
    latitudes = compute_gaussian_latitudes(n_lat, iters=iters, tol=tol)
    
    return latitudes, longitudes 
end

"""
    Compute Gaussian latitudes using Newton's method to find roots of Legendre polynomials.
    Returns latitudes in radians, sorted from North to South (decreasing order).
    If n_lat is a common value (32 or 64), precomputed roots are returned to ensure consistency with precomputed available QG3 data.
"""
function compute_gaussian_latitudes(n_lat::Int; iters::Int=100, tol::Real=1e-10)
    #for common n_lat values, return precomputed roots for QG3 compatibility
    if n_lat == 32
        return [1.4967943, 1.4009757, 1.3046336, 1.2079424, 1.1114256, 1.0147344, 0.9182177, 0.82152647, 0.7248352, 0.62831855, 
        0.5316273, 0.43493605, 0.3382448, 0.24155357, 0.14503686, 0.048345618, -0.048345618, -0.14503686, -0.24155357, -0.3382448, 
        -0.43493605, -0.5316273, -0.62831855, -0.7248352, -0.82152647, -0.9182177, -1.0147344, -1.1114256, -1.2079424, -1.3046336, -1.4009757, -1.4967943]
    elseif n_lat == 64
        return [1.5335126, 1.4852146, 1.4366313, 1.3879837, 1.3393116, 1.2906276, 1.241937, 1.1932425, 1.1445451, 1.095846, 1.0471455, 
        0.9984441, 0.94974184, 0.901039, 0.8523357, 0.8036321, 0.7549281, 0.7062239, 0.6575194, 0.6088149, 0.5601101, 0.51140517, 0.4627002,
         0.41399515, 0.36529, 0.31658477, 0.2678795, 0.21917419, 0.17046884, 0.121763475, 0.0730581, 0.0243527, -0.0243527, -0.0730581, -0.121763475,
          -0.17046884, -0.21917419, -0.2678795, -0.31658477, -0.36529, -0.41399515, -0.4627002, -0.51140517, -0.5601101, -0.6088149, -0.6575194, -0.7062239,
           -0.7549281, -0.8036321, -0.8523357, -0.901039, -0.94974184, -0.9984441, -1.0471455, -1.095846, -1.1445451, -1.1932425, -1.241937, -1.2906276,
            -1.3393116, -1.3879837, -1.4366313, -1.4852146, -1.5335126]
    else
        n = n_lat
        x_values = zeros(Float64, n)
        
        for i in 1:n
            #initial guess
            k = Float64(i)
            x0 = cos(π * (k - 0.25 + 1/(8*(2k-1))) / (n + 0.5))
            
            x = x0
            for iter in 1:iters
                p, dp = legendre_polynomial(n, x)
                if abs(p) < tol
                    break
                end
                delta = p / dp
                x_new = x - delta
                if x_new < -1.0 || x_new > 1.0
                    x_new = x - 0.5 * delta
                end
                x = x_new
            end
            x_values[i] = x
        end
        
        # Sort and convert to latitudes
        return asin.(sort(x_values, rev=true))
    end
end

"""
    Compute Legendre polynomial P_n(x) and its derivative using recurrence relation.
    This uses the standard normalization where P_n(1) = 1.
"""
function legendre_polynomial(n::Int, x::Float64)
    if n == 0
        return 1.0, 0.0
    elseif n == 1
        return x, 1.0
    end
    
    p_prev = 1.0
    p_curr = x
    dp_prev = 0.0
    dp_curr = 1.0
    
    for k in 2:n
        # Recurrence relation for Legendre polynomials
        p_next = ((2k - 1) * x * p_curr - (k - 1) * p_prev) / k
        dp_next = ((2k - 1) * (p_curr + x * dp_curr) - (k - 1) * dp_prev) / k
        
        p_prev, p_curr = p_curr, p_next
        dp_prev, dp_curr = dp_curr, dp_next
    end
    
    return p_curr, dp_curr
end

"""
    qg3pars_constructor_helper(L::Int, n_lat::Int; n_lon::Int=2*n_lat, iters::Int=100, tol::Real=1e-8,NF::Type{<:AbstractFloat}=Float32)

Helper function to hook the constructor for QG3ModelParameters using a Gaussian grid.
Generates latitude/longitude points and initializes empty topography and land/sea mask.
Used mainly to handle SH transforms.

# Arguments

-`L::Int`: Spectral truncation level (maximum degree).

- `n_lat::Int`: Number of Gaussian latitudes.

# Keywords

- `n_lon::Int=2*n_lat`: Number of longitudes (default: twice the latitude count).

- `iters::Int=100`: Maximum number of iterations for Gaussian grid convergence.

- `tol::Real=1e-8`: Convergence tolerance.

- `NF::Type{<:AbstractFloat}=Float32`: Number format for outputs.

# Returns

- `QG3ModelParameters`: Model parameters including grid coordinates, topography (h),
and land-sea mask (LS).

# Example
```julia
pars = qg3pars_constructor_helper(42, 64)
```
"""
function qg3pars_constructor_helper(L::Int, n_lat::Int; n_lon::Int=2*n_lat, iters::Int=100, tol::Real=1e-8,NF::Type{<:AbstractFloat}=Float32)
    @assert L <= n_lat "L must be less than or equal to n_lat"
    lats, lons  = (gaussian_grid(n_lat; n_lon=n_lon, iters=iters, tol=tol))
    lats, lons = NF.(lats), NF.(lons)
    LS = h = zeros(NF, n_lat, n_lon)
    QG3ModelParameters(L, lats, lons, LS, h)
end

function qg3pars_constructor_helper(L::Int, qg3ppars::QG3.QG3ModelParameters; NF::Type{<:AbstractFloat}=Float32)
    @assert L <= qg3ppars.N_lats "L must be less than or equal to n_lat"
    _, lons  = (gaussian_grid(qg3ppars.N_lats; n_lon=qg3ppars.N_lons))
    lats = qg3ppars.lats
    lats, lons = NF.(lats), NF.(lons)
    LS = h = zeros(NF, qg3ppars.N_lats, qg3ppars.N_lons)
    QG3ModelParameters(L, lats, lons, LS, h)
end
