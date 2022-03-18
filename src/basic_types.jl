# This file holds the basic that store the precomputed model and grid information

import Base.show, Base.eltype

"""
    QG3ModelParameters{T}

Saves all the parameters of the model

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
    * `a::T` Earth's radius  # WIP: some functions assume a=1
    * `Ω::T` Earth's angular speed for Coriolis # WIP: some functions assume Ω=1/2
    * `gridtype::String` either 'gaussian' or 'regular', 'gaussian' leads to all transform being done by somewhat naively implemented transforms, 'regular' uses FastTransforms.jl
    * `time_unit::T` unit so that t [normalized] = t [s] / time_unit, default = 1/4π
    * `distance_unit::T`, default Earth's radius
    * `ψ_unit::T`, derived from time_unit and distance_unit
    * `q_unit::T`, derived from time_unit and distance_unit

# Other constructors

    QG3ModelParameters(L::Int, lats::AbstractArray{T,1}, lons::AbstractArray{T,1}, LS::AbstractArray{T,2}, h::AbstractArray{T,2}, R1i::T, R2i::T, H0::T, τRi::T, τEi::T, τHi::T, α1::T, α2::T, a::T, Ω::T, gridtype::String, time_unit::T, distance_unit::T)

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
    a::T
    Ω::T

    gridtype::String

    time_unit::T
    distance_unit::T
    ψ_unit::T
    q_unit::T
end

function QG3ModelParameters(L::Int, lats::AbstractArray{T,1}, lons::AbstractArray{T,1}, LS::AbstractArray{T,2}, h::AbstractArray{T,2}, R1i::Number=82.83600204081633, R2i::Number=200.44267160493825, H0::Number=9000., τRi::Number=0.0031830988618379067, τEi::Number=0.026525823848649224, τHi::Number=0.039788735772973836, α1::Number=0.5, α2::Number=0.5, a::Number=1., Ω::Number=1/2, gridtype::String="gaussian", time_unit::Number=0.07957747154594767, distance_unit::Number=6.371e6) where T<:Real

    N_lats = size(lats,1)

    M = 2*L - 1
    N_lons = size(lons,1)

    colats = lat_to_colat.(lats)
    μ = sin.(lats)

    λmax = (length(lats)-1) * ((length(lats)-1) +1) # largest eigenvalue of the laplacian of the grid (in spherical harmonics)
    cH = τHi * a^8 * (λmax)^(-4)

    ψ_unit = (distance_unit*distance_unit)/(24*60*60*time_unit)
    q_unit = 1/(60*60*24*time_unit)

    return QG3ModelParameters(L, M, N_lats, N_lons, lats, colats, μ, lons, LS, h, T(R1i), T(R2i), T(H0), T(τRi), T(τEi), T(cH), T(α1), T(α2), T(a), T(Ω), gridtype, T(time_unit), T(distance_unit), T(ψ_unit), T(q_unit))
end

togpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, togpu(p.lats), togpu(p.θ), togpu(p.μ), togpu(p.lons), togpu(p.LS), togpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.a, p.Ω, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

tocpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, tocpu(p.lats), tocpu(p.θ), tocpu(p.μ), tocpu(p.lons), tocpu(p.LS), tocpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.a, p.Ω, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

show(io::IO, p::QG3ModelParameters{T}) where {T} = print(io," QG3ModelParameters{",T,"} with N_lats=",p.N_lats," N_lons=",p.N_lons," L_max=",p.L-1)

"""
     GaussianGrid{T, onGPU} <: AbstractGridType{T, onGPU}

Struct for the transforms and derivates of a Gaussian Grid

# Fields:

* `GtoSH::GaussianGridtoSHTransform`
* `SHtoG::GaussianSHtoGridTransform`
* `dμ::GaussianGrid_dμ`
* `dλ::Derivative_dλ`
"""
struct GaussianGrid{T, onGPU} <: AbstractGridType{T, onGPU}
    GtoSH
    SHtoG
    dμ
    dλ
    size_SH
    size_grid
end

show(io::IO, g::GaussianGrid{T, true}) where {T} = print(io," Gaussian Grid on GPU")
show(io::IO, g::GaussianGrid{T, false}) where {T} = print(io," Gaussian Grid on CPU")

"""
    grid(p::QG3ModelParameters{T})

Convience constructor for the [`AbstractGridType`](@ref) based on the parameters set in `p`.
"""
function grid(p::QG3ModelParameters{T}, gridtype::String, N_level::Int=3) where T<:Number

    if gridtype=="regular"
        
        error("Not implemented anymore, as aliasing problems couldn't be fixed, there is some code in the old commits of the rep though.")

    elseif gridtype=="gaussian"

        GtoSH = GaussianGridtoSHTransform(p, N_level)
        SHtoG = SHtoGaussianGridTransform(p, N_level)
        dμ = GaussianGrid_dμ(p, N_level)
        dλ = Derivative_dλ(p)

        size_grid = (p.N_lats, p.N_lons)
        size_SH = cuda_used[] ? (p.N_lats, p.N_lons+2) : (p.L, p.M)

        return GaussianGrid{T, cuda_used[]}(GtoSH, SHtoG, dμ, dλ, size_SH, size_grid)
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
* `Δ::AbstractArray{T,2}` # laplace operator in spherical harmonical coordinates
* `Tψq` matrix used for transforming stream function to voriticy   q = Tψq * ψ + F
* `Tqψ``inverse of Tψq   ψ = Tqψ*(q - F)
* `f` modified coriolis vector used in transforming stream function to vorticity
* `J_f3` coriolis contribution to Jacobian at 850hPa
* `cH∇8` cH * 8-th order gradient for horizonatal diffusion
* `cH∇8_3d` cH * 8-th order gradient for horizonatal diffusion for 3d field
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
    Δ::AbstractArray{T,2}
    Δ_3d
    Tψq
    Tqψ
    f
    f_J3
    cH∇8
    cH∇8_3d
    ∂k∂ϕ
    ∂k∂μ
    ∂k∂λ
end

"""
    QG3Model(p::QG3ModelParameters)

Routine that pre computes the QG3 Model and returns a QG3Model struct with all precomputed fields except for the forcing.

The pre-computation is always done on CPU due to scalar indexing being used, if a GPU is avaible the final result is then transferred to the GPU.
"""
function QG3Model(p::QG3ModelParameters)

    p = togpu(p)

    k = togpu(compute_k(p))
    cosϕ = togpu(compute_cosϕ(p))
    acosϕi = togpu(compute_acosϕi(p))

    g = grid(p)

    Δ = cuda_used[] ? reorder_SH_gpu(compute_Δ(p), p) : compute_Δ(p)

    Tqψ, Tψq = compute_batched_ψq_transform_matrices(p, Δ)
    Tqψ, Tψq = togpu(Tqψ), togpu(Tψq)

    TRcoeffs = togpu(compute_TR(p))
    f = transform_SH(togpu(compute_coriolis_vector_grid(p)), g)

    TR_matrix = togpu(compute_batched_TR_matrix(p))
    f_J3 = togpu(compute_f_J3(p, f))

    ∇8 = cuda_used[] ? reorder_SH_gpu(compute_∇8(p), p) : compute_∇8(p)
    ∇8 *= p.cH
    
    k_SH = transform_SH(k, g)

    ∂k∂μ = SHtoGrid_dμ(k_SH, g.dμ)
    ∂k∂λ = transform_grid(SHtoSH_dφ(k_SH, g.dλ), g) ./ (cosϕ .^ 2)
    ∂k∂ϕ = SHtoGrid_dϕ(k_SH, g.dμ)

    return QG3Model(p, g, k, TRcoeffs, TR_matrix, cosϕ, acosϕi, Δ, make3d(Δ), Tψq, Tqψ, f, f_J3, ∇8, make3d(∇8), ∂k∂ϕ, ∂k∂μ, ∂k∂λ)
end

show(io::IO, m::QG3Model{T}) where {T} = print(io, "Pre-computed QG3Model{",T,"} with ",m.p, " on a",m.g, "with gridsize ",m.g.size_SH," and L_max",m.p.L - 1," on ", isongpu_string(m)) 

"""
    isongpu(m::QG3Model{T})

Determines if the model was pre-computed to be used on GPU.
"""
isongpu(m::QG3Model{T}) where {T} = typeof(m.g) <: AbstractGridType{T, true}

isongpu_string(m::QG3Model{T}) where {T} = isongpu(m::QG3Model{T}) ? "GPU" : "CPU"

eltype(m::QG3Model{T}) where {T} = T
