import Base.show 

# this file contains all the code to take derivatives

# derivate functions follow the naming scheme: "Domain1Input"to"Domain2Output"_d"derivativeby"

# there are variants for GPU and CPU, the different grids and 3d and 2d fields

abstract type AbstractDerivative{onGPU} end 

abstract type AbstractλDerivative{onGPU} <: AbstractDerivative{onGPU} end

"""
abstract type AbstractμDerivative{onGPU} < AbstractDerivative{onGPU} 

Required fields: 

* `msinθ`: To change between μ and latitude derivative (-sin(colats))
* `msinθ_3d`: To change between μ and latitude derivative (-sin(colats))

"""
abstract type AbstractμDerivative{onGPU} <: AbstractDerivative{onGPU} end

"""
derivative of input after θ (azimutal angle/colatitude) in SH, uses pre computed SH evaluations (dependend on the grid type)
"""
SHtoGrid_dθ(ψ, m::QG3Model{T}) where {T} = SHtoGrid_dθ(ψ, m.g.dμ)

SHtoGrid_dθ(ψ::AbstractArray{T,2}, d::AbstractμDerivative) where {T} = d.msinθ .* SHtoGrid_dμ(ψ, d)

SHtoGrid_dθ(ψ::AbstractArray{T,3}, d::AbstractμDerivative) where {T} = d.msinθ_3d .* SHtoGrid_dμ(ψ, d)

"""
    Derivative_dλ

Pre-computes Derivatives by longitude. Uses the SH relation, is therefore independ from the grid.
"""
struct Derivative_dλ{R,S,T,onGPU} <: AbstractλDerivative{onGPU}
    mm::R
    mm_3d::S
    swap_m_sign_array::T
end

function Derivative_dλ(p::QG3ModelParameters{T}) where {T}

    mm = mMatrix(p)
    mm_3d = make3d(mm)

    if cuda_used[]
        mm = reorder_SH_gpu(mm, p)
        mm_3d = reorder_SH_gpu(mm_3d, p)
        swap_m_sign_array = cu([1; Int((p.N_lons)/2)+3 : p.N_lons + 2; 1:Int((p.N_lons)/2)+1;])
    else 
        swap_m_sign_array = [1;vcat([[2*i+1,2*i] for i=1:p.L-1]...)]
    end

    Derivative_dλ{typeof(mm), typeof(mm_3d), typeof(swap_m_sign_array), cuda_used[]}(mm, mm_3d, swap_m_sign_array)
end

"""
derivative of input after φ (polar angle) or λ (longtitude) in SH to Grid, only for a single layer
"""
SHtoGrid_dφ(ψ, m::QG3Model{T}) where {T} = transform_grid(SHtoSH_dφ(ψ,m), m)

SHtoGrid_dλ(ψ, m) = SHtoGrid_dφ(ψ, m)
SHtoSH_dλ(ψ, m) = SHtoSH_dφ(ψ, m)

"""
derivative of input after φ (polar angle/longtitude) in SH, output in SH
"""
SHtoSH_dφ(ψ, m::QG3Model{T}) where {T} = SHtoSH_dφ(ψ, m.g)
SHtoSH_dφ(ψ, g::GaussianGrid) = SHtoSH_dφ(ψ, g.dλ)


# 2d field variant
SHtoSH_dφ(ψ::AbstractArray{T,2}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm, g.swap_m_sign_array)

# 3d field variant
SHtoSH_dφ(ψ::AbstractArray{T,3}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm_3d, g.swap_m_sign_array)

_SHtoSH_dφ(ψ::AbstractArray{T,N}, mm::AbstractArray{T,N}, swap_arr) where {T,N} = mm .* change_msign(ψ, swap_arr)


"""
    change_msign(A)

Change the sign of the m in SH. This version returns a view

there is currently a bug or at least missing feature in Zygote, the AD library, that stops views from always working flawlessly when a view is mixed with prior indexing of an array. We need a view for the derivative after φ to change the sign of m, so here is a differentiable variant of the SHtoSH_dφ function for the 2d field
"""
change_msign(A::AbstractArray{T,2}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,swap_array)

# 3d field version
change_msign(A::AbstractArray{T,3}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,:,swap_array)

Zygote.@adjoint function change_msign(A::AbstractArray{T,N}, swap_array::AbstractArray{Int,1}) where {T,N}
    return (change_msign(A,swap_array), Δ->(change_msign(Δ,swap_array),nothing))
end

change_msign(A::AbstractArray{T,3}, i::Integer, swap_array::AbstractArray{Int,1}) where T<:Number = @inbounds view(A,i,:,swap_array)

"""
    GaussianGrid_dμ(p::QG3ModelParameters{T}, N_level::Int=3)

Pre-computes Pseudo-spectral approach to computing derivatives with repsect to μ = sin(lats). Derivatives are called with following the naming scheme: "Domain1Input"to"Domain2Output"_d"derivativeby"

"""
struct GaussianGrid_dμ{onGPU} <: AbstractμDerivative{onGPU}
    t::SHtoGaussianGridTransform
    msinθ
    msinθ_3d  
end

show(io::IO, t::GaussianGrid_dμ{true}) = print(io, "Pre-computed SH to Gaussian Grid Derivative{",P,"} on GPU")
show(io::IO, t::GaussianGrid_dμ{false}) = print(io, "Pre-computed SH to Gaussian Grid Derivative{",P,"} on CPU")

function GaussianGrid_dμ(p::QG3ModelParameters{T}, N_level::Int=3) where {T}
    dPμdμ, __, __ = compute_P(p)
    A_real = togpu(rand(T, N_level, p.N_lats, p.N_lons))

    if cuda_used[]
        dPμdμ = reorder_SH_gpu(dPμdμ, p)

        FT_2d = plan_cur2r(A_real[1,:,:], 2)
        iFT_2d = plan_cuir2r(FT_2d*(A_real[1,:,:]), p.N_lons, 2)

        FT_3d = plan_cur2r(A_real, 3)
        iFT_3d = plan_cuir2r(FT_3d*A_real, p.N_lons, 3)

    else 
        iFT_2d = FFTW.plan_r2r(A_real[1,:,:], FFTW.HC2R, 2)
        iFT_3d = FFTW.plan_r2r(A_real, FFTW.HC2R, 3)

    end
    outputsize = (p.N_lats, p.N_lons)

    msinθ = togpu(T.(reshape(-sin.(p.θ),p.N_lats, 1)))
    msinθ_3d = togpu(make3d(msinθ))

    GaussianGrid_dμ{cuda_used[]}(SHtoGaussianGridTransform{T,typeof(iFT_2d),typeof(iFT_3d),typeof(dPμdμ),cuda_used[]}(iFT_2d, iFT_3d, dPμdμ, outputsize, p.N_lats, p.N_lons, p.M), msinθ, msinθ_3d)
end

"""
derivative of input after μ = sinϕ in SH, uses pre computed SH evaluations
"""
SHtoGrid_dμ(ψ, m::QG3Model{T}) where {T}= SHtoGrid_dμ(ψ, m.g)
SHtoGrid_dμ(ψ, g::AbstractGridType) = SHtoGrid_dμ(ψ, g.dμ)
SHtoGrid_dμ(ψ, d::GaussianGrid_dμ) = transform_grid(ψ, d.t)

SHtoSH_dθ(ψ, m) = transform_SH(SHtoGrid_dθ(ψ,m), m)
SHtoSH_dϕ(ψ, m) = eltype(ψ)(-1) .* SHtoSH_dθ(ψ, m)
SHtoGrid_dϕ(ψ, m) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, m)
SHtoGrid_dϕ(ψ, d::GaussianGrid_dμ) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, d)
