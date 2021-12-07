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

    return QG3ModelParameters(L, M, N_lats, N_lons, lats, colats, μ, lons, LS, h, R1i, R2i, H0, τRi, τEi, cH, α1, α2, a, Ω, gridtype, time_unit, distance_unit, ψ_unit, q_unit)
end

togpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, togpu(p.lats), togpu(p.θ), togpu(p.μ), togpu(p.lons), togpu(p.LS), togpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.a, p.Ω, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

tocpu(p::QG3ModelParameters) = QG3ModelParameters(p.L, p.M, p.N_lats, p.N_lons, tocpu(p.lats), tocpu(p.θ), tocpu(p.μ), tocpu(p.lons), tocpu(p.LS), tocpu(p.h), p.R1i, p.R2i, p.H0, p.τRi, p.τEi, p.cH, p.α1, p.α2, p.a, p.Ω, p.gridtype, p.time_unit, p.distance_unit, p.ψ_unit, p.q_unit)

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
    msinθ::AbstractArray{T,2}
    set_spurious_zero::Bool # set spurious modes zero
    mm::AbstractArray{T,2} # (-m) SH number matrix, used for zonal derivative
    mm_3d::AbstractArray{T,3} # (-m) SH number matrix, used for zonal derivative for 3d fields
    swap_m_sign_array # indexing array, used to swap the sign of the m SH number, used for zonal derivative
end

"""
     GaussianGrid{T, onGPU} <: AbstractGridType{T, onGPU}

Struct for Gaussian grid and its transforms.

# Fields:

* `P::AbstractArray{T,3}` ass. Legendre Polynomials
* `Pw::AbstractArray{T,3}` ass. Legendre Polynomials * Gaussian weights
* `FT` Fourier transform plan
* `iFT` inverse Fourier transform plan
* `FT_3d` Fourier transform plan for fully matrix version with lvl as first dimension
* `iFT_3d` inverse Fourier transform plan for fully matrix version with lvl as first dimension
* `truncate_array` truncatation indices
* `dPμdμ::AbstractArray{T,3}` derivative of ass. Legendre Polynomials
* `msinθ::AbstractArray{T,2}` -sin(θ) for dervative with repsect to θ and ϕ
* `mm::AbstractArray{T,2}` (-m) SH number matrix, used for zonal derivative
* `mm_3d::AbstractArray{T,3}` (-m) SH number matrix, used for zonal derivative for 3d fields
* `swap_m_sign_array` indexing array, used to swap the sign of the m SH number, used for zonal derivative

"""
struct GaussianGrid{T, onGPU} <: AbstractGridType{T, onGPU}
    P::AbstractArray{T,3}
    Pw::AbstractArray{T,3}
    FT
    iFT
    FT_3d
    iFT_3d
    truncate_array
    dPμdμ::AbstractArray{T,3}
    msinθ::AbstractArray{T,2}
    msinθ_3d::AbstractArray{T,3}
    mm::AbstractArray{T,2}
    mm_3d::AbstractArray{T,3}
    swap_m_sign_array
end

"""
    grid(p::QG3ModelParameters{T})

Convience constructor for the [`AbstractGridType`](@ref) based on the parameters set in `p`.
"""
function grid(p::QG3ModelParameters{T}, gridtype::String) where T<:Number

    dPμdμ, __, P = compute_P(p)
    A_real = togpu(rand(T,3, p.N_lats, p.N_lons))

    mm = compute_mmMatrix(p)
    mm_3d = make3d(mm)
    swap_m_sign_array = [1;vcat([[2*i+1,2*i] for i=1:p.L-1]...)]

    msinθ = togpu(zeros(T, p.N_lats, p.N_lons))
    for i ∈ 1:p.N_lats
        msinθ[i,:] .= -sin(p.θ[i])
    end
    msinθ_3d = make3d(msinθ)

    if gridtype=="regular"
        SH = plan_sph2fourier(T, p.N_lats)
        FT = plan_sph_analysis(T, p.N_lats, p.N_lons)
        FTinv = plan_sph_synthesis(T, p.N_lats, p.N_lons)
        ∂_iFT = FFTW.plan_r2r(A_real, FFTW.HC2R, 2)

        P_spurious_modes = togpu(prepare_sph_zero_spurious_modes(p))
        setzero

        return RegularGrid(SH, FT, FTinv, ∂_iFT, P_spurious_modes, CS, dPμdμ, msinθ, msinθ_3d, true, mm, mm_3d, swap_m_sign_array)

    elseif gridtype=="gaussian"

        Pw = deepcopy(P)
        Pw = compute_LegendreGauss(p, Pw)

        if cuda_used[]

            mm = reorder_SH_gpu(mm, p)
            mm_3d = reorder_SH_gpu(mm_3d, p)
            swap_m_sign_array = [1; Int((p.N_lons)/2)+3 : p.N_lons + 2; 1:Int((p.N_lons)/2)+1;]

            P, Pw, dPμdμ = reorder_SH_gpu(P, p), reorder_SH_gpu(Pw, p), reorder_SH_gpu(dPμdμ, p)

            _FT = CUDA.CUFFT.plan_rfft(A_real[1,:,:], 2)
            _iFT = CUDA.CUFFT.plan_irfft((FT*A_real[1,:,:]), p.N_lons, 2)
            # also set up the inverse plans for the adjoints, this is not done automatically by CUDA.jl
            _FT.pinv = CUDA.CUFFT.plan_inv(_FT)
            _iFT.pinv = CUDA.CUFFT.plan_inv(_iFT)

            # these work with AD (see gpu_r2r_transform.jl)
            FT = plan_cur2r(_FT, 2)
            iFT = plan_icur2r(_iFT, p.N_lons, 2)

            _FT_3d = CUDA.CUFFT.plan_rfft(A_real, 3)
            _iFT_3d = CUDA.CUFFT.plan_irfft((FT_3d*A_real), p.N_lons, 3)

            _FT_3d.pinv = CUDA.CUFFT.plan_inv(_FT_3d)
            _iFT_3d.pinv = CUDA.CUFFT.plan_inv(_iFT_3d)

            FT = plan_cur2r(_FT_3d, 3)
            iFT = plan_icur2r(_iFT_3d, p.N_lons, 3)

            truncate_array = nothing
        else
            FT = FFTW.plan_r2r(A_real[1,:,:], FFTW.R2HC, 2)
            iFT = FFTW.plan_r2r(A_real[1,:,:], FFTW.HC2R, 2)

            FT_3d = FFTW.plan_r2r(A_real, FFTW.R2HC, 3)
            iFT_3d = FFTW.plan_r2r(A_real, FFTW.HC2R, 3)

            m_p = 1:p.L
            m_n = p.N_lons:-1:p.N_lons-(p.L-2)

            truncate_array = [1]
            for im=1:(p.L-1)
                push!(truncate_array, m_n[im])
                push!(truncate_array, m_p[im+1])
            end
        end

        return GaussianGrid{T, cuda_used[]}(togpu(P), togpu(Pw), FT, iFT, FT_3d, iFT_3d, truncate_array, togpu(dPμdμ), togpu(msinθ), togpu(msinθ_3d), togpu(mm), togpu(mm_3d), togpu(swap_m_sign_array))
    else
        error("Unknown gridtype.")
    end
end
grid(p::QG3ModelParameters) = grid(p, p.gridtype)

abstract type AbstractQG3Model{T} end

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
* `∇8` 8-th order gradient for horizonatal diffusion
* `∇8_3d` 8-th order gradient for horizonatal diffusion for 3d field
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
    Tψq
    Tqψ
    f
    f_J3
    ∇8
    ∇8_3d
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
    f = transform_SH(togpu(compute_coriolis_vector_grid(p)), p, g)

    TR_matrix = togpu(compute_batched_TR_matrix(p))
    f_J3 = togpu(compute_f_J3(p, f))

    ∇8 = cuda_used[] ? reorder_SH_gpu(compute_∇8(p), p) : compute_∇8(p)

    k_SH = transform_SH(k, p, g)

    ∂k∂μ = SHtoGrid_dμ(k_SH, p, g)
    ∂k∂λ = transform_grid(SHtoSH_dφ(k_SH, g), p, g) ./ (cosϕ .^ 2)
    ∂k∂ϕ = SHtoGrid_dϕ(k_SH, p, g)

    return QG3Model(p, g, k, TRcoeffs, TR_matrix, cosϕ, acosϕi, Δ, Tψq, Tqψ, f, f_J3, ∇8, make3d(∇8), ∂k∂ϕ, ∂k∂μ, ∂k∂λ)
end

"""
    isongpu(m::QG3Model{T})

Determines if the model was pre-computed to be used on GPU.
"""
isongpu(m::QG3Model{T}) where T<:Number = typeof(m.g) <: AbstractGridType{T, true}
