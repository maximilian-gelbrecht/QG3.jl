import Base.show 

# this file contains all the code to take derivatives

# derivate functions follow the naming scheme: "Domain1Input"to"Domain2Output"_d"derivativeby"

# there are variants for GPU and CPU, the different grids and 3d and 2d fields

# longitude derivates are computed via SH relation 
# latitude derivatives are computed pseudo-spectral with pre-computed ass. legendre polynomials

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
    Derivative_dλ(p::QG3ModelParameters{T}; N_batch::Int=0)

Pre-computes Derivatives by longitude. Uses the SH relation, is therefore independ from the grid.
"""
struct Derivative_dλ{R,S,A4,T,onGPU} <: AbstractλDerivative{onGPU}
    mm::R
    mm_3d::S
    mm_4d::A4
    swap_m_sign_array::T
end

function Derivative_dλ(p::QG3ModelParameters{T}; N_batch::Int=0) where {T}

    mm = -(mMatrix(p))
    mm_3d = make3d(mm)

    if cuda_used[]
        mm = reorder_SH_gpu(mm, p)
        mm_3d = reorder_SH_gpu(mm_3d, p)
        swap_m_sign_array = cu([1; Int((p.N_lons)/2)+3 : p.N_lons + 2; 1:Int((p.N_lons)/2)+1;])
    else 
        swap_m_sign_array = [1;vcat([[2*i+1,2*i] for i=1:p.L-1]...)]
    end

    if N_batch > 0 
        mm_4d = repeat(mm_3d, 1,1,1,1)
    else 
        mm_4d = nothing
    end

    Derivative_dλ{typeof(mm), typeof(mm_3d), typeof(mm_4d), typeof(swap_m_sign_array), cuda_used[]}(mm, mm_3d, mm_4d, swap_m_sign_array)
end

"""
    SHtoGrid_dφ(ψ, m::QG3Model{T})
    SHtoGrid_dφ(ψ, m::AbstractGridType{T})
    SHtoGrid_dφ(ψ, dλ::Derivative_dλ, sh2g::AbstractSHtoGridTransform)

Derivative of input after φ (polar angle) or λ (longtitude) in SH to Grid
"""
SHtoGrid_dφ(ψ, m::QG3Model{T}) where {T} = transform_grid(SHtoSH_dφ(ψ,m), m)
SHtoGrid_dφ(ψ, g::AbstractGridType{T}) where {T} = transform_grid(SHtoSH_dφ(ψ,g), g)
SHtoGrid_dφ(ψ, dλ::Derivative_dλ, sh2g::AbstractSHtoGridTransform) = transform_grid(SHtoSH_dφ(ψ,dλ), sh2g)

"""
    SHtoGrid_dλ(ψ, m::QG3Model{T})
    SHtoGrid_dλ(ψ, m::AbstractGridType{T})

Derivative of input after φ (polar angle) or λ (longtitude) in SH to Grid
"""
SHtoGrid_dλ(ψ, m) = SHtoGrid_dφ(ψ, m)
SHtoGrid_dλ(ψ, dl, sh2g) = SHtoGrid_dφ(ψ, dl, sh2g)

"""
    SHtoSH_dλ(ψ, m::QG3Model{T})
    SHtoSH_dλ(ψ, g::GaussianGrid) 

Derivative of input after φ (polar angle/longtitude) in SH, output in SH
"""
SHtoSH_dλ(ψ, m) = SHtoSH_dφ(ψ, m)

"""
    SHtoSH_dφ(ψ, m::QG3Model{T})
    SHtoSH_dφ(ψ, g::GaussianGrid) 
    SHtoSH_dφ(ψ, g::Derivative_dλ) 

Derivative of input after φ (polar angle/longtitude) in SH, output in SH
"""
SHtoSH_dφ(ψ, m::QG3Model{T}) where {T} = SHtoSH_dφ(ψ, m.g)
SHtoSH_dφ(ψ, g::GaussianGrid) = SHtoSH_dφ(ψ, g.dλ)

# 2d field variant
SHtoSH_dφ(ψ::AbstractArray{T,2}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm, g.swap_m_sign_array)

# 3d field variant
SHtoSH_dφ(ψ::AbstractArray{T,3}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm_3d, g.swap_m_sign_array)

# 4d field variant 
SHtoSH_dφ(ψ::AbstractArray{T,4}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm_4d, g.swap_m_sign_array)


_SHtoSH_dφ(ψ::AbstractArray{T,N}, mm::AbstractArray{T,N}, swap_arr) where {T,N} = mm .* change_msign(ψ, swap_arr)


"""
    change_msign(A)

Change the sign of the m in SH. This version returns a view

there is currently a bug or at least missing feature in Zygote, the AD library, that stops views from always working flawlessly when a view is mixed with prior indexing of an array. We need a view for the derivative after φ to change the sign of m, so here is a differentiable variant of the SHtoSH_dφ function for the 2d field
"""
change_msign(A::AbstractArray{T,2}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,swap_array)

# 3d field version
change_msign(A::AbstractArray{T,3}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,:,swap_array)

# 4d field version 
change_msign(A::AbstractArray{T,4}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,:,swap_array,:)


Zygote.@adjoint function change_msign(A::AbstractArray{T,N}, swap_array::AbstractArray{Int,1}) where {T,N}
    return (change_msign(A,swap_array), Δ->(change_msign(Δ,swap_array),nothing))
end

change_msign(A::AbstractArray{T,3}, i::Integer, swap_array::AbstractArray{Int,1}) where T<:Number = @inbounds view(A,i,:,swap_array)

"""
    GaussianGrid_dμ(p::QG3ModelParameters{T}, N_level::Int=3; N_batch::Int=0)

Pre-computes Pseudo-spectral approach to computing derivatives with repsect to μ = sin(lats). Derivatives are called with following the naming scheme: "Domain1Input"to"Domain2Output"_d"derivativeby"

"""
struct GaussianGrid_dμ{onGPU, S<:SHtoGaussianGridTransform, M, A, A4} <: AbstractμDerivative{onGPU}
    t::S
    msinθ::M
    msinθ_3d::A
    msinθ_4d::A4
end

show(io::IO, t::GaussianGrid_dμ{true}) = print(io, "Pre-computed SH to Gaussian Grid Derivative on GPU")
show(io::IO, t::GaussianGrid_dμ{false}) = print(io, "Pre-computed SH to Gaussian Grid Derivative on CPU")

function GaussianGrid_dμ(p::QG3ModelParameters{T}, N_level::Int=3; N_batch::Int=0) where {T}
    dPμdμ, __ = compute_P(p)
    A_real = togpu(rand(T, N_level, p.N_lats, p.N_lons))

    if N_batch > 0 
        A_real4d = togpu(rand(T, N_level, p.N_lats, p.N_lons, N_batch))
    else 
        iFT_4d = nothing 
    end  

    if cuda_used[]
        dPμdμ = reorder_SH_gpu(dPμdμ, p)

        FT_2d = plan_r2r_AD(A_real[1,:,:], 2)
        iFT_2d = plan_ir2r_AD(FT_2d*(A_real[1,:,:]), p.N_lons, 2)

        FT_3d = plan_r2r_AD(A_real, 3)
        iFT_3d = plan_ir2r_AD(FT_3d*A_real, p.N_lons, 3)

        if N_batch > 0 
            FT_4d = plan_r2r_AD(A_real4d, 3)
            iFT_4d = plan_ir2r_AD(FT_4d*A_real4d, p.N_lons, 3)
        end 
    else 
        iFT_2d = plan_ir2r_AD(A_real[1,:,:], p.N_lons, 2)
        iFT_3d = plan_ir2r_AD(A_real, p.N_lons, 3)

        if N_batch > 0 
            iFT_4d = plan_ir2r_AD(A_real4d, p.N_lons, 3)
        end 
    end
    outputsize = (p.N_lats, p.N_lons)

    msinθ = togpu(T.(reshape(-sin.(p.θ),p.N_lats, 1)))
    msinθ_3d = togpu(make3d(msinθ))
    
    if N_batch > 0 
        msinθ_4d = repeat(msinθ_3d, 1,1,1,1)
    else 
        msinθ_4d = nothing 
    end

    transform = SHtoGaussianGridTransform{T, typeof(iFT_2d), typeof(iFT_3d), typeof(iFT_4d), typeof(dPμdμ), typeof(outputsize), typeof(p.N_lats), cuda_used[]}(iFT_2d, iFT_3d, iFT_4d, dPμdμ, outputsize, p.N_lats, p.N_lons, p.M)
    
    GaussianGrid_dμ{cuda_used[], typeof(transform), typeof(msinθ), typeof(msinθ_3d), typeof(msinθ_4d)}(transform, msinθ, msinθ_3d, msinθ_4d)
end

"""
    SHtoGrid_dμ(ψ, m::QG3Model{T}) 
    SHtoGrid_dμ(ψ, g::AbstractGridType) 
    SHtoGrid_dμ(ψ, d::GaussianGrid_dμ) 

Derivative of input after μ = sinϕ in SH, uses pre computed SH evaluations
"""
SHtoGrid_dμ(ψ, m::QG3Model{T}) where {T}= SHtoGrid_dμ(ψ, m.g)
SHtoGrid_dμ(ψ, g::AbstractGridType) = SHtoGrid_dμ(ψ, g.dμ)
SHtoGrid_dμ(ψ, d::GaussianGrid_dμ) = transform_grid(ψ, d.t)

SHtoSH_dθ(ψ, m) = transform_SH(SHtoGrid_dθ(ψ,m), m)
SHtoSH_dϕ(ψ, m) = eltype(ψ)(-1) .* SHtoSH_dθ(ψ, m)

"""
    SHtoGrid_dϕ(ψ, m)

Derivative of input after ϕ - latitude in SH, uses pre computed SH evaluations
"""
SHtoGrid_dϕ(ψ, m) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, m)
SHtoGrid_dϕ(ψ, d::GaussianGrid_dμ) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, d)

"""
derivative of input after θ (azimutal angle/colatitude) in SH, uses pre computed SH evaluations (dependend on the grid type)
"""
SHtoGrid_dθ(ψ, m::QG3Model{T}) where {T} = SHtoGrid_dθ(ψ, m.g.dμ)
SHtoGrid_dθ(ψ, g::GaussianGrid{T}) where {T} = SHtoGrid_dθ(ψ, g.dμ)
SHtoGrid_dθ(ψ::AbstractArray{T,2}, d::AbstractμDerivative) where {T} = d.msinθ .* SHtoGrid_dμ(ψ, d)
SHtoGrid_dθ(ψ::AbstractArray{T,3}, d::AbstractμDerivative) where {T} = d.msinθ_3d .* SHtoGrid_dμ(ψ, d)
SHtoGrid_dθ(ψ::AbstractArray{T,4}, d::AbstractμDerivative) where {T} = d.msinθ_4d .* SHtoGrid_dμ(ψ, d)

"""
    Laplacian(p::QG3ModelParameters{T}; init_inverse=false, R::T=T(1), N_batch::Int=0, kwargs...) where T

Initializes the `Laplacian` in spherical harmonics and if `init_inverse==true` also its inverse

Apply the Laplacian with the functions (@ref)[`Δ`] and (@ref)[`Δ⁻¹`]
"""
struct Laplacian{T,M<:AbstractArray{T,2},A1<:AbstractArray{T,3},A2<:AbstractArray{T,3},A4,A5,onGPU} <: AbstractDerivative{onGPU}
    Δ::M
    Δ_3d::A1
    Δ_4d::A4
    Δ⁻¹::M
    Δ⁻¹_3d::A2
    Δ⁻¹_4d::A5
end 

function Laplacian(p::QG3ModelParameters{T}; init_inverse=false, R::T=T(1), N_batch::Int=0, kwargs...) where T
    
    Δ = cuda_used[] ? reorder_SH_gpu(compute_Δ(p), p) : compute_Δ(p)
    Δ ./= (R*R)
    Δ_3d = make3d(Δ)

    if init_inverse
        Δ⁻¹ = cuda_used[] ? reorder_SH_gpu(compute_Δ⁻¹(p), p) : compute_Δ⁻¹(p)
        Δ⁻¹ .*= (R*R)
        Δ⁻¹_3d = make3d(Δ⁻¹)
    else 
        Δ⁻¹ = Array{T,2}(undef,0,0)
        Δ⁻¹_3d = Array{T,3}(undef,0,0,0)
    end 
        
    if N_batch > 0 
        Δ_4d = repeat(Δ_3d, 1,1,1,1)
        Δ⁻¹_4d = repeat(Δ⁻¹_3d, 1,1,1,1)  
    else 
        Δ_4d = nothing
        Δ⁻¹_4d = nothing  
    end

    Laplacian{T, typeof(Δ), typeof(Δ_3d), typeof(Δ⁻¹_3d), typeof(Δ_4d), typeof(Δ⁻¹_4d), cuda_used[]}(Δ, Δ_3d, Δ_4d, Δ⁻¹, Δ⁻¹_3d, Δ⁻¹_4d)
end 

"""
    Δ(ψ::AbstractArray, L::Laplacian{T})
    Δ(ψ::AbstractArray, m::QG3Model{T})

Apply the Laplacian. Also serves to convert regular vorticity (not the quasigeostrophic one) to streamfunction) 
"""
Δ(ψ::AbstractArray{T,4}, L::Laplacian{T}) where T = L.Δ_4d .* ψ
Δ(ψ::AbstractArray{T,3}, L::Laplacian{T}) where T = L.Δ_3d .* ψ
Δ(ψ::AbstractArray{T,2}, L::Laplacian{T}) where T = L.Δ .* ψ

Δ(ψ::AbstractArray{T,N}, g::AbstractGridType{T}) where {T,N} = Δ(ψ, g.Δ)
Δ(ψ::AbstractArray{T,N}, m::QG3Model{T}) where {T,N} = Δ(ψ, m.g)

"""
    Δ⁻¹(ψ::AbstractArray, m::QG3Model{T})

Apply the inverse Laplacian. Also serves to convert the streamfunction to regular vorticity 
"""
Δ⁻¹(ψ::AbstractArray{T,4}, L::Laplacian{T}) where T = L.Δ⁻¹_4d .* ψ
Δ⁻¹(ψ::AbstractArray{T,3}, L::Laplacian{T}) where T = L.Δ⁻¹_3d .* ψ
Δ⁻¹(ψ::AbstractArray{T,2}, L::Laplacian{T}) where T = L.Δ⁻¹ .* ψ 

Δ⁻¹(ψ::AbstractArray{T,N}, g::AbstractGridType{T}) where {T,N} = Δ⁻¹(ψ, g.Δ)
Δ⁻¹(ψ::AbstractArray{T,N}, m::QG3Model{T}) where {T,N} = Δ⁻¹(ψ, m.g) 

"""
    Hyperdiffusion(p::QG3ModelParameters{T}; scale::T=T(1)) where T

Initializes the Hyperdiffusion / horizonatal diffusion operator. 

Apply it via the (@ref)[`∇8`] functions.
"""
struct Hyperdiffusion{T, M<:AbstractArray{T,2}, A<:AbstractArray{T,3}, A4, onGPU} <: AbstractDerivative{onGPU}
    ∇8::M
    ∇8_3d::A
    ∇8_4d::A4
end 

function Hyperdiffusion(p::QG3ModelParameters{T}; hyperdiffusion_scale::T=T(1), N_batch::Int=0, kwargs...) where T

    ∇8 = cuda_used[] ? reorder_SH_gpu(compute_∇8(p), p) : compute_∇8(p)
    ∇8 .*= hyperdiffusion_scale 

    ∇8_3d = make3d(∇8) 
    if N_batch > 0 
        ∇8_4d = repeat(∇8_3d, 1,1,1,1)
    else 
        ∇8_4d = nothing 
    end 

    Hyperdiffusion{T, typeof(∇8), typeof(∇8_3d), typeof(∇8_4d), cuda_used[]}(∇8, ∇8_3d, ∇8_4d)
end 

"""
    ∇8(q::AbstractArray{T,N}, H::Hyperdiffusion{T})

Apply the hyperdiffusion to the input
"""
∇8(q::AbstractArray{T,2}, H::Hyperdiffusion{T}) where T = H.∇8 .* q
∇8(q::AbstractArray{T,3}, H::Hyperdiffusion{T}) where T = H.∇8_3d .* q
∇8(q::AbstractArray{T,4}, H::Hyperdiffusion{T}) where T = H.∇8_4d .* q
∇8(q::AbstractArray{T,N}, g::AbstractGridType{T}) where {T,N} = ∇8(q, g.∇8)
∇8(q::AbstractArray{T,N}, m::QG3Model{T}) where {T,N} = ∇8(q, m.g)

cH∇8(varargs...) = ∇8(varargs...)