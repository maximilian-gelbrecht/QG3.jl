# This file holds the basic that store the precomputed model and grid information

import Base.show, Base.eltype

"""
    QG3ModelParameters{T}

Saves all the parameters of the model, also the units/normalization. 

# A word on the normalization of the model

The model is normalized in (natural) units of the system. 

* Distance: The Earth's radius is set to `1` (and thus all units containing distance are scaled with the actual radius of the Earth)
* Vorticity: Earth's angular speed is approximately 2π/d, here Earth's angular speed is set 2Ω = 1, so that the planetery vorticity component of the Jacobian is just ∂ψ/∂λ. Therefore: Ω = 1/2, it follows 2*2π [1/d] = 1 [Ω_normalized], therefore Ω_normalized = 4π/d. 
* Time: From the vorticity unit, follows the time unit is [d/4π] = [d_normalized]. Hence, time in days must be devided by 4π to get the natural time unit, or [d_normalized] = 4π [d]
* both ψ and q thus need to expressed in these units too, this can be done using the `ψ_unit` and `q_unit`

# Fields

    * `L::Int`: number of SH number l, note that 0,...,l_max, so that L is l_max + 1
    * `M::Int`: # maximum number M values, M==2*L - 1 / 2*l_max + 1
    * `N_lats::Int`
    * `N_lons::Int`
    * `lats::AbstractArray{T,1}`
    * `θ::AbstractArray{T,1}` colatitudes = θ
    * `μ::AbstractArray{T,1}` sin(lats) == cos(colats)
    * `LS::AbstractArray{T,2}` Land see mask, on the same grid as lats and lons
    * `h::AbstractArray{T,2}`` orography, array on the same grid as lats and lons
    * `R1i::T` square inverse of Rossby radius 1 200-500hPa, default 1/(700km)^2
    * `R2i::T` square inverse of Rossby radius 2 500-800hPa, default 1/(450km)^2
    * `H0::T` scaling parameter for topography, default 9000m
    * `τRi::T` inverse of radiative time scale, default 1/(25d)
    * `τEi::T` inverse of drag time scale, default 1/(3d)
    * `cH::T` horizonatal diffusion coefficient
    * `α1::T` drag coefficient 1, default 0.5
    * `α2::T` drag coefficient 2, default 0.5
    * `gridtype::String` either 'gaussian' or 'regular', 'gaussian' leads to all transform being done by somewhat naively implemented transforms, 'regular' uses FastTransforms.jl
    * `time_unit::T` unit so that t [normalized] = t [d] / time_unit, default = 1 / 4π
    * `distance_unit::T`, default Earth's radius, so that s [normalized] = s [m] / distance_unit, default = 6.371e6
    * `ψ_unit::T`, derived from time_unit and distance_unit, so that ψ [normalized] = ψ [m^2/s] / ψ_unit
    * `q_unit::T`, derived from time_unit and distance_unit

# Other constructors

    QG3ModelParameters(L::Int, lats::AbstractArray{T,1}, lons::AbstractArray{T,1}, LS::AbstractArray{T,2}, h::AbstractArray{T,2}, R1i::T, R2i::T, H0::T, τRi::T, τEi::T, τHi::T, α1::T, α2::T, a::T, gridtype::String, time_unit::T, distance_unit::T)

Default arguments for most parameters, so that

    QG3ModelParameters(L::Int, lats::AbstractArray{T,1}, lons::AbstractArray{T,1}, LS::AbstractArray{T,2}, h::AbstractArray{T,2})

works as well.
"""
struct QG3ModelParameters{T}
    L::Int
    M::Int
    N_lats::Int
    N_lons::Int

    lats::AbstractArray{T,1}
    θ::AbstractArray{T,1}
    μ::AbstractArray{T,1}

    lons::AbstractArray{T,1}

    LS::AbstractArray{T,2}
    h::AbstractArray{T,2}

    R1i::T
    R2i::T
    H0::T

    τRi::T
    τEi::T
    cH::T
    α1::T
    α2::T

    gridtype::String

    time_unit::T
    distance_unit::T
    ψ_unit::T
    q_unit::T
end

function QG3ModelParameters(L::Int, lats::AbstractArray{T,1}, lons::AbstractArray{T,1}, LS::AbstractArray{T,2}=zeros(T,(1,1)), h::AbstractArray{T,2}=zeros(T,(1,1)), R1i::Number=82.83600204081633, R2i::Number=200.44267160493825, H0::Number=9000., τRi::Number=(1/25)/(4π), τEi::Number=(1/3)/(4π), τHi::Number=(1/2)/(4π), α1::Number=0.5, α2::Number=0.5, gridtype::String="gaussian", time_unit::Number=T(1/4π), distance_unit::Number=6.371e6) where T<:Real

    N_lats = size(lats,1)

    M = 2*L - 1
    N_lons = size(lons,1)

    colats = lat_to_colat.(lats)
    μ = sin.(lats)

    λmax = (L-1) * (L) # largest eigenvalue of the laplacian of the grid (in spherical harmonics)
    cH = τHi * (λmax)^(-4) # * a^8, but a==1

    ψ_unit = (distance_unit*distance_unit)/(24*60*60*time_unit)
    q_unit = 1/(60*60*24*time_unit)

    return QG3ModelParameters(L, M, N_lats, N_lons, lats, colats, μ, lons, LS, h, T(R1i), T(R2i), T(H0), T(τRi), T(τEi), T(cH), T(α1), T(α2), gridtype, T(time_unit), T(distance_unit), T(ψ_unit), T(q_unit))
end

togpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, togpu(p.lats), togpu(p.θ), togpu(p.μ), togpu(p.lons), togpu(p.LS), togpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

tocpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, tocpu(p.lats), tocpu(p.θ), tocpu(p.μ), tocpu(p.lons), tocpu(p.LS), tocpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

show(io::IO, p::QG3ModelParameters{T}) where {T} = print(io," QG3ModelParameters{",T,"} with N_lats=",p.N_lats," N_lons=",p.N_lons," L_max=",p.L-1)

"""
     GaussianGrid{T, onGPU} <: AbstractGridType{T, onGPU}

Struct for the transforms and derivates of a Gaussian Grid

# Fields:

* `GtoSH::GaussianGridtoSHTransform`
* `SHtoG::GaussianSHtoGridTransform`
* `dμ::GaussianGrid_dμ`
* `dλ::Derivative_dλ`
* `Δ::Laplacian`
* `∇8::Hyperdiffusion`
"""
struct GaussianGrid{T, onGPU} <: AbstractGridType{T, onGPU}
    GtoSH
    SHtoG
    dμ
    dλ
    Δ
    ∇8
    size_SH
    size_grid
end

show(io::IO, g::GaussianGrid{T, true}) where {T} = print(io," Gaussian Grid on GPU")
show(io::IO, g::GaussianGrid{T, false}) where {T} = print(io," Gaussian Grid on CPU")

"""
    grid(p::QG3ModelParameters{T})

Convience constructor for the [`AbstractGridType`](@ref) based on the parameters set in `p`.
"""
function grid(p::QG3ModelParameters{T}, gridtype::String, N_level::Int=3; kwargs...) where T<:Number

    if gridtype=="regular"
        
        error("Not implemented anymore, as aliasing problems couldn't be fixed, there is some code in the old commits of the rep though.")

    elseif gridtype=="gaussian"

        GtoSH = GaussianGridtoSHTransform(p, N_level)
        SHtoG = SHtoGaussianGridTransform(p, N_level)
        dμ = GaussianGrid_dμ(p, N_level)
        dλ = Derivative_dλ(p)
        Δ = Laplacian(p; kwargs...)
        ∇8 = Hyperdiffusion(p; kwargs...)

        size_grid = (p.N_lats, p.N_lons)
        size_SH = cuda_used[] ? (p.N_lats, p.N_lons+2) : (p.L, p.M)

        return GaussianGrid{T, cuda_used[]}(GtoSH, SHtoG, dμ, dλ,  Δ, ∇8, size_SH, size_grid)
    else
        error("Unknown gridtype.")
    end
end
grid(p::QG3ModelParameters) = grid(p, p.gridtype)

"""
    QG3Model{T}

Holds all parameter and grid information, plus additional pre-computed fields that save computation time during model integration. All these parameter are seen as constants and not trainable. It should be computed on CPU, if you are on GPU use `CUDA.@allowscalar QG3Model(...)`.

# Fields

* `p::QG3ModelParameters{T}` Parameters of the model
* `g::AbstractGridType{T}` Grid type with pre-computed plans or Legendre polynomals and co.
* `k::AbstractArray{T,2}` Array, drag coefficient pre-computed from orography and land-sea mask
* `TRcoeffs::AbstractArray{T,3}` Array of Temperature relaxation coefficients (2d version)
* `TR_matrix` Matrix for temperature relaxation (3d version)
* `cosϕ::AbstractArray{T,2}` cos(latitude) pre computed
* `acosϕi::AbstractArray{T,2}` inverse of a*cos(latitude) pre computed
* `Tψq` matrix used for transforming stream function to voriticy   q = Tψq * ψ + F
* `Tqψ``inverse of Tψq   ψ = Tqψ*(q - F)
* `f` modified coriolis vector used in transforming stream function to vorticity
* `f_J3` coriolis contribution to Jacobian at 850hPa
* `∂k∂ϕ` derivates of drag coefficients, pre computed for Ekman dissipation
* `∂k∂μ`
* `∂k∂λ` includes 1/cos^2ϕ (for Ekman dissipiation computation)

"""
struct QG3Model{T} <: AbstractQG3Model{T}
    p::QG3ModelParameters{T}
    g::AbstractGridType{T}
    k::AbstractArray{T,2}
    TRcoeffs::AbstractArray{T,3}
    TR_matrix
    cosϕ::AbstractArray{T,2}
    acosϕi::AbstractArray{T,2}
    Tψq
    Tqψ
    f
    f_J3
    ∂k∂ϕ
    ∂k∂μ
    ∂k∂λ
end

"""
    QG3Model(p::QG3ModelParameters; N_level::Integer=3, kwargs...)

Routine that pre computes the QG3 Model and returns a QG3Model struct with all precomputed fields except for the forcing.

The pre-computation is always done on CPU due to scalar indexing being used, if a GPU is avaible the final result is then transferred to the GPU.

If `N_level`` is set to values other than three, the pre-computations for the transforms and derivatives are computed with N_level levels, but the full model will not be working! This is just for accessing the transforms and derivatives

All ``kwargs`` are forwarded to `grid`.
"""
function QG3Model(p::QG3ModelParameters; N_levels::Integer=3, kwargs...)
    
    p = togpu(p)

    k = togpu(compute_k(p))
    cosϕ = togpu(compute_cosϕ(p))
    acosϕi = togpu(compute_acosϕi(p))

    if "hyperdiffusion_scale" in keys(kwargs)
        g = grid(p, p.gridtype, N_levels; kwargs...)
    else 
        g = grid(p, p.gridtype, N_levels; hyperdiffusion_scale=p.cH, kwargs...)
    end

    if N_levels != 3
        @warn "QG3.jl: N_level is not set to 3, the full model will not be functioning!" 

        # we need a temporary grid during the initialization in that case currently
        if "hyperdiffusion_scale" in keys(kwargs)
            grid_temp = grid(p, p.gridtype, 3; kwargs...)
        else 
            grid_temp = grid(p, p.gridtype, 3; hyperdiffusion_scale=p.cH, kwargs...)
        end
    
        grid_temp = grid(p, p.gridtype, 3)
    else 
        grid_temp = g 
    end

    Tqψ, Tψq = compute_batched_ψq_transform_matrices(p, grid_temp.Δ)
    Tqψ, Tψq = togpu(Tqψ), togpu(Tψq)

    TRcoeffs = togpu(compute_TR(p))
    f = transform_SH(togpu(compute_coriolis_vector_grid(p)), grid_temp)

    TR_matrix = togpu(compute_batched_TR_matrix(p))
    f_J3 = togpu(compute_f_J3(p, f))
    
    k_SH = transform_SH(k, grid_temp)

    ∂k∂μ = SHtoGrid_dμ(k_SH, grid_temp.dμ)
    ∂k∂λ = transform_grid(SHtoSH_dφ(k_SH, g.dλ), grid_temp) ./ (cosϕ .^ 2)
    ∂k∂ϕ = SHtoGrid_dϕ(k_SH, grid_temp.dμ)

    grid_temp = nothing

    return QG3Model(p, g, k, TRcoeffs, TR_matrix, cosϕ, acosϕi, Tψq, Tqψ, f, f_J3, ∂k∂ϕ, ∂k∂μ, ∂k∂λ)
end

show(io::IO, m::QG3Model{T}) where {T} = print(io, "Pre-computed QG3Model{",T,"} with ",m.p, " on a",m.g, "with gridsize ",m.g.size_SH," and L_max",m.p.L - 1," on ", isongpu_string(m)) 

"""
    isongpu(m::QG3Model{T})

Determines if the model was pre-computed to be used on GPU.
"""
isongpu(m::QG3Model{T}) where {T} = typeof(m.g) <: AbstractGridType{T, true}

isongpu_string(m::QG3Model{T}) where {T} = isongpu(m::QG3Model{T}) ? "GPU" : "CPU"

eltype(m::QG3Model{T}) where {T} = T
