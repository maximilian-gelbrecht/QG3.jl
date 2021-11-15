
"""
    QG3ModelParameters{T}

Saves all the parameters of the model

# Fields

    * `L::Int`: number of SH number l, note that 0,...,l_max, so that L is l_max + 1
    * `M::Int`: # maximum number M values, M==2*L - 1 / 2*l_max + 1
    * `N_lats::Int`
    * `N_lons::Int`
    * `lats::AbstractArray{T,1}`
    * `colats::AbstractArray{T,1}` colatitudes = θ
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
    colats::AbstractArray{T,1}
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

function QG3ModelParameters(L::Int, lats::AbstractArray{T,1}, lons::AbstractArray{T,1}, LS::AbstractArray{T,2}, h::AbstractArray{T,2}, R1i::T=82.83600204081633, R2i::T=200.44267160493825, H0::T=9000., τRi::T=0.0031830988618379067, τEi::T=0.026525823848649224, τHi::T=0.039788735772973836, α1::T=0.5, α2::T=0.5, a::T=1., Ω::T=1/2, gridtype::String="gaussian", time_unit::T=0.07957747154594767, distance_unit::T=6.371e6) where T<:Real

    N_lats = size(lats,1)

    M = 2*L - 1
    N_lons = size(lons,1)

    colats = lat_to_colat(lats)
    μ = sin.(lats)

    λmax = (length(lats)-1) * ((length(lats)-1) +1) # largest eigenvalue of the laplacian of the grid (in spherical harmonics)
    cH = τHi * a^8 * (λmax)^(-4)

    ψ_unit = (distance_unit*distance_unit)/(24*60*60*time_unit)
    q_unit = 1/(60*60*24*time_unit)

    return QG3ModelParameters(L, M, N_lats, N_lons, togpu(lats), togpu(colats), togpu(μ), togpu(lons), togpu(LS), togpu(h), R1i, R2i, H0, τRi, τEi, cH, α1, α2, a, Ω, gridtype, time_unit, distance_unit, ψ_unit, q_unit)
end

togpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, togpu(p.lats), togpu(p.colats), togpu(p.μ), togpu(p.lons), togpu(p.LS), togpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.a, p.Ω, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

tocpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, tocpu(p.lats), tocpu(p.colats), tocpu(p.μ), tocpu(p.lons), tocpu(p.LS), tocpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.a, p.Ω, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

"""
    AbstractGridType{T, onGPU}

Abstract type for grids. The grids save information about the transform from the spatial to spectral grid, e.g. pre-computed Legendre Polynomials
"""
abstract type AbstractGridType{T, onGPU} end

struct RegularGrid{T, onGPU} <: AbstractGridType{T, onGPU}
    SH # plan for sph2fourier transform "\" is transform to SH and "*" the inverse transform
    FT # plan for bivariate fourier transform
    FTinv  # plan for inverse bivariate fourier transform
    ∂_iFT # for derivative
    P_spurious_modes # mask, used to zero spurious SH modes
    CS::AbstractArray{T,2}
    dPμdμ::AbstractArray{T,3}
    dPcosθdθ::AbstractArray{T,3}
    set_spurious_zero::Bool # set spurious modes zero
end

struct GaussianGrid{T, onGPU} <: AbstractGridType{T, onGPU}
    P::AbstractArray{T,3} # ass. Legendre Polynomials
    Pw::AbstractArray{T,3} # ass. Legendre Polynomials * Gaussian weights
    FT # Fourier transform plan
    iFT # inverse Fourier transform plan
    truncate_array # truncatation indices
    dPμdμ::AbstractArray{T,3} # derivative of ass. Legendre Polynomials
    dPcosθdθ::AbstractArray{T,3} # derivative of ass. Legendre Polynomials
end

function grid(p::QG3ModelParameters{T}, gridtype::String) where T<:Number

    dPμdμ, dPcosθdθ, P = compute_P(p)
    A_real = togpu(rand(T,p.N_lats, p.N_lons))

    if gridtype=="regular"
        SH = plan_sph2fourier(T, p.N_lats)
        FT = plan_sph_analysis(T, p.N_lats, p.N_lons)
        FTinv = plan_sph_synthesis(T, p.N_lats, p.N_lons)
        ∂_iFT = FFTW.plan_r2r(A_real, FFTW.HC2R, 2)

        P_spurious_modes = togpu(prepare_sph_zero_spurious_modes(p))
        setzero

        return RegularGrid(SH, FT, FTinv, ∂_iFT, togpu(P_spurious_modes), togpu(CS), togpu(dPμdμ), togpu(dPcosθdθ), true)

    elseif gridtype=="gaussian"
        SH = nothing
        FT = nothing
        FTinv = nothing

        Pw = deepcopy(P)
        Pw = compute_LegendreGauss(p, Pw)

        if cuda_used
            FT = CUDA.CUFFT.plan_rfft(A_real, 2)
            iFT = CUDA.CUFFT.plan_irfft((FT*A_real), p.N_lons, 2)

            truncate_array = [1]

            for im=1:(p.L-1)
                push!(truncate_array, p.L+im) # Imag part
                push!(truncate_array, im+1) # Real part
            end
        else
            FT = FFTW.plan_r2r(A_real, FFTW.R2HC, 2)
            iFT = FFTW.plan_r2r(A_real, FFTW.HC2R, 2)

            m_p = 1:p.L
            m_n = p.N_lons:-1:p.N_lons-(p.L-2)

            truncate_array = [1]
            for im=1:(p.L-1)
                push!(truncate_array, m_n[im])
                push!(truncate_array, m_p[im+1])
            end
        end

        return GaussianGrid{T, cuda_used}(togpu(P), togpu(Pw), FT, iFT, togpu(truncate_array), togpu(dPμdμ), togpu(dPcosθdθ))
    else
        error("Unknown gridtype.")
    end
end
grid(p::QG3ModelParameters) = grid(p, p.gridtype)

"""
    QG3Model{T}

Holds all parameter and grid information, plus additional pre-computed fields that save computation time during model integration
"""
struct QG3Model{T}
    # all these parameter are seen as constants and not trainable, depending on the gridtype some may be 'nothing' as they are only needed for one of the grid types / transform variants
    p::QG3ModelParameters{T}
    g::AbstractGridType{T} # Grid type with pre-computed plans or Legendre polynomals and co.

    k::AbstractArray{T,2} # Array, drag coefficient pre-computed from orography and land-sea mask
    TRcoeffs::AbstractArray{T,3} # Array of Temperature relaxation coefficients

    cosϕ::AbstractArray{T,2} # cos(latitude) pre computed
    acosϕi::AbstractArray{T,2} # inverse of a*cos(latitude) pre computed
    Δ::AbstractArray{T,2} # laplace operator in spherical harmonical coordinates
    mm::AbstractArray{T,2} # (-m) SH number matrix, used for zonal derivative
    swap_m_sign_array # indexing array, used to swap the sign of the m SH number, used for zonal derivative

    Tψq # matrix used for transforming stream function to voriticy   q = Tψq * ψ + F
    Tqψ # inverse of Tψq   ψ = Tqψ*(q - F)
    f # modified coriolis vector used in transforming stream function to vorticity
    ∇8 # 8-th order gradient for horizonatal diffusion

    ∂k∂ϕ # derivates of drag coefficients, pre computed , ∂k∂ϕ includes 1/cos^2ϕ
    ∂k∂μ
    ∂k∂λ
end


"""
    QG3Model(p::QG3ModelParameters)

Routine that pre computes the QG3 Model and returns a QG3Model struct with all precomputed fields except for the forcing.
"""
function QG3Model(p::QG3ModelParameters)
    k = compute_k(p)
    cosϕ = compute_cosϕ(p)
    acosϕi = compute_acosϕi(p)

    g = grid(p)

    Δ = compute_Δ(p)
    mm = compute_mmMatrix(p)
    swap_m_sign_array = [1;vcat([[2*i+1,2*i] for i=1:p.L-1]...)]

    Tqψ, Tψq = compute_batched_ψq_transform_matrices(p, Δ)
    TRcoeffs = compute_TR(p)

    f = transform_SH(togpu(compute_coriolis_vector_grid(p)), p, g)
    ∇8 = compute_∇8(p)

    k_SH = transform_SH(togpu(k), p, g)

    ∂k∂μ = SHtoGrid_dμ(k_SH, p, g)
    ∂k∂λ = transform_grid(_SHtoSH_dφ(k_SH, togpu(mm), togpu(swap_m_sign_array)), p, g) ./ togpu((cosϕ .^ 2))
    ∂k∂ϕ = SHtoGrid_dϕ(k_SH, p, g)

    return QG3Model(p, g, togpu(k), togpu(TRcoeffs), togpu(cosϕ), togpu(acosϕi), togpu(Δ), togpu(mm), togpu(swap_m_sign_array), togpu(Tψq), togpu(Tqψ), togpu(f), togpu(∇8), togpu(∂k∂ϕ), togpu(∂k∂μ), togpu(∂k∂λ))
end
