
# q and ψ are always three dimensional, no matter if in real space (level,lon/lat), fourier (level,kx,ky) or spherical (level,il,m)

# right now the equations is solved in real spherical harmonics expansion

# The transform is handled either by naive SH transform or the FastTransforms.jl library and is pre-computed. The FastTransforms.jl is currently invoking aliasing problems and not working for the full model. All SH are handled in the matrix convention that FastTransforms.jl uses: columns by m-value: 0, -1, 1, -2, 2, ..., rows l in ascending order. This is for the naive SH transform definately not the fasted way of storing the coefficients as an additonal allocating reordering needs to be done for every transform. Therefore the coefficient matrix convention might be changed in future versions of this model.

# the whole code is written to be differentiable by Zygote. This is why all function are written in a non-mutating way, this is slightly slower on CPU, on GPU some of these functions like batched matrix multicplication should be faster. Even when a GPU is detected, a lot of the pre-computation is done on CPU, the integration however is performed on GPU

# right now it is a bit unconsitant with weather normailization with a==1 \Omega==1 is enforced or not


"""
Convert the streamfunction ψ to (anomlous) potential vorticity q' in spherical harmonics basis

This version is slightly slower than the old one on CPU (as it not aware of the matrix being half full of zeroes), but it is non-mutating which makes it suitable for the automatic differentation.

It replaces the double loop over the coefficient matrix with a batched vector multiply. The advantage other besides it being non-mutating is that it is optimised for GPU, so it might actually be faster on the GPU than doing a manual loop.
"""
function ψtoqprime(p::QG3Model{T}, ψ::AbstractArray{T,3}) where T<:Number
    return reshape(batched_vec(p.Tψq, reshape(ψ,3,:)),3 , p.p.L, p.p.M)
end

"""
Convert the streamfunction ψ to the potential vorticity
"""
ψtoq(p::QG3Model{T}, ψ::AbstractArray{T,3}) where T<:Number = ψtoqprime(p, ψ) + p.f

"""
Convert the potential vorticity q to streamfunction ψ
"""
qtoψ(p::QG3Model{T}, q::AbstractArray{T,3}) where T<:Number = qprimetoψ(p, q - p.f)

"""
Convert the anomalous potential vorticity q' to streamfunction ψ

This version is slightly slower than the old one (as it not aware of the matrix being half full of zeroes), but it is non-mutating which makes it suitable for the automatic differentation.

It replaces the double loop over the coefficient matrix with a batched vector multiply. The advantage of that is that it is optimised for GPU, so it might actually be faster on the GPU than doing a manual loop.
"""
function qprimetoψ(p::QG3Model{T}, q::AbstractArray{T,3}) where T<:Number
    return reshape(batched_vec(p.Tqψ, reshape(q,3,:)),3 , p.p.L, p.p.M)
end


"""
Compute the Jacobian determinant from ψ and q in μ,λ coordinates, J = ∂ψ/∂x ∂q/∂y - ∂ψ/∂y ∂q/∂x = 1/a^2cosϕ ( - ∂ψ/∂λ ∂q/∂ϕ + ∂ψ/∂ϕ ∂q/∂λ) =  1/a^2 (- ∂ψ/∂λ ∂q/∂μ + ∂ψ/∂μ ∂q/∂λ)

The last term ∂ψ/∂λ accounts for the planetery vorticity, actually it is 2Ω ∂ψ/∂λ, but 2Ω == 1, (write q = q' + 2Ωμ to proof it)

"""
J(ψ, q, m::QG3Model) = transform_SH(SHtoGrid_dμ(ψ, m).*SHtoGrid_dλ(q, m) - (SHtoGrid_dλ(ψ, m).*SHtoGrid_dμ(q, m)), m) - SHtoSH_dλ(ψ, m)

"""
Compute the Jacobian determinant from ψ and q in μ,λ coordinates without the planetary vorticity, as used in computing the eddy/transient forcing
"""
J_F(ψ, q, m::QG3Model) = transform_SH(SHtoGrid_dμ(ψ, m).*SHtoGrid_dλ(q, m) - (SHtoGrid_dλ(ψ, m).*SHtoGrid_dμ(q, m)), m)

"""
For the Jacobian at 850hPa, q = q' + f(1+h/H_0) = q' + f + f*h/H_0, so that the thrid term has to be added.
"""
J3(ψ, q, m::QG3Model) = J(ψ, q + (m.f[3,:,:] - m.f[2,:,:]), m)


"""
Ekman dissipation

 EK = ∇(k∇ψ) = (∇k ∇ψ) + k Δψ
 EK = 1/a^2cos^2ϕ ∂k/∂λ ∂ψ/∂λ + 1/a^2 ∂k/∂ϕ ∂ψ/∂ϕ + k Δψ   (a==1)

 m.∂k∂λ  includes 1/cos^2ϕ
"""
EK(ψ::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = transform_SH(SHtoGrid_dϕ(ψ[3,:,:], m) .* m.∂k∂ϕ + SHtoGrid_dλ(ψ[3,:,:], m) .* m.∂k∂λ + m.k .* transform_grid(m.Δ .* ψ[3,:,:], m), m)


"""
Simplified Ekman dissipiation for k(ϕ,λ) = const
"""
EK3_simple(ψ::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = m.p.τEi .* m.Δ .* ψ[3,:,:]


D1(ψ::AbstractArray{T,3}, qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = -TR12(m, ψ) + H(qprime, 1, m)
D2(ψ::AbstractArray{T,3}, qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = TR12(m,ψ ) - TR23(m, ψ) + H(qprime, 2, m)
D3(ψ::AbstractArray{T,3}, qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = TR23(m, ψ) + EK(ψ, m) + H(qprime, 3, m)

D3_simple(ψ::AbstractArray{T,3}, qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = TR23(m, ψ) + EK3_simple(ψ, m) + H(qprime, 3, m)

"""
Temperature relaxation
"""
TR(m::QG3Model{T}, ψ1, ψ2, Ri::T) where T<: Number = m.p.τRi .* Ri .* (ψ1 - ψ2)
TR12(m::QG3Model{T}, ψ::AbstractArray{T,3}) where T<:Number = m.TRcoeffs[1,:,:] .* (ψ[1,:,:] - ψ[2,:,:])
TR23(m::QG3Model{T}, ψ::AbstractArray{T,3}) where T<:Number = m.TRcoeffs[2,:,:] .* (ψ[2,:,:] - ψ[3,:,:])

"""
Horizontal diffusion, q' is anomolous pv (without coriolis)
"""
H(qprime::AbstractArray{T,3}, i::Int, m::QG3Model{T}) where T<: Number = m.p.cH .* (m.∇8 .* qprime[i,:,:])

u(ψ, m) = -m.p.a^(-1) .* SHtoGrid_dϕ(ψ, m)
v(ψ, m) = m.acosϕi .* SHtoGrid_dλ(ψ, m)


"""
derivative of input after φ (polar angle) or λ (longtitude) in SH to Grid, only for a single layer
"""
SHtoGrid_dφ(ψ::AbstractArray{T,2}, m::QG3Model{T}) where T<:Number = transform_grid(SHtoSH_dφ(ψ,m), m)
SHtoGrid_dφ(ψ::AbstractArray{T,3}, i::Integer, m::QG3Model{T}) where T<:Number = transform_grid(SHtoSH_dφ(ψ, i, m), m)
SHtoGrid_dλ(ψ, m) = SHtoGrid_dφ(ψ, m)

"""
derivative of input after φ (polar angle/longtitude) in SH, output in SH
"""
SHtoSH_dφ(ψ::AbstractArray{T,2}, m::QG3Model{T}) where T<:Number = _SHtoSH_dφ(ψ, m.mm, m.swap_m_sign_array)

_SHtoSH_dφ(ψ::AbstractArray{T,2}, mm::AbstractArray{T,2}, swap_arr) where T<:Number = mm .* change_msign(ψ, swap_arr)

"""
derivative of input after φ (polar angle/longtitude) in SH, output in SH

these are  variants with AbstractArray{T,3} and index i to select which layer is the input for the derivative.

there is currently a bug or at least missing feature in Zygote, the AD library, that stops views from always working flawlessly when a view is mixed with prior indexing of an array. We need a view for the derivative after φ to change the sign of m, so here is a differentiable variant of the SHtoSH_dφ function
"""
SHtoSH_dφ(ψ::AbstractArray{T,3}, i::Integer, m::QG3Model{T}) where T<:Number = _SHtoSH_dφ(ψ, i, m.mm)

_SHtoSH_dφ(ψ::AbstractArray{T,3}, i::Integer, mm::AbstractArray{T,2}) where T<:Number = mm .* change_msign(ψ, i)

SHtoSH_dλ(ψ, m) = SHtoSH_dφ(ψ, m)


"""
derivative of input after θ (azimutal angle/colatitude) in SH, uses pre computed SH evaluations (dependend on the grid type)
"""
SHtoGrid_dθ(ψ::AbstractArray{T,2}, m::QG3Model{T}) where T<:Number = SHtoGrid_dθ(ψ, m.p, m.g)

SHtoGrid_dθ(ψ::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T}) where T<:Number = _SHtoGrid_dμθ(ψ, g.dPcosθdθ, p, g)

SHtoGrid_dθ(ψ::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::RegularGrid{T}) where T<:Number = _SHtoGrid_dμθ(ψ, g.dPcosθdθ, p, g)

"""
derivative of input after μ = sinϕ in SH, uses pre computed SH evaluations
"""
SHtoGrid_dμ(ψ::AbstractArray{T,2}, m::QG3Model{T}) where T<:Number = SHtoGrid_dμ(ψ, m.p, m.g)

SHtoGrid_dμ(ψ::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T}) where T<:Number = _SHtoGrid_dμθ(ψ, g.dPμdμ, p, g)

SHtoGrid_dμ(ψ::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::RegularGrid{T}) where T<:Number = _SHtoGrid_dμθ(ψ, g.dPμdμ, p, g)

"""
Performs the latitude-based derivates. Uses a form of synthesis based on pre-computed values of the ass. Legendre polynomials at the grid points.

This version is optimized to be non-mutating with batched_vec. The inner batched_vec correspodends to an inverse Legendre transform and the outer multiply to an inverse Fourier transform.
"""
function _SHtoGrid_dμθ(ψ::AbstractArray{T,2}, dP::AbstractArray{T,3}, p::QG3ModelParameters, g::AbstractGridType{T, false}) where T<:Number

    out = batched_vec(dP, ψ)

    g.iFT * cat(out[:,1:2:end], zeros(T, p.N_lats, p.N_lons - p.M), out[:,end-1:-2:2], dims=2)
end

function _SHtoGrid_dμθ(ψ::AbstractArray{T,2}, dP::AbstractArray{T,3}, p::QG3ModelParameters, g::AbstractGridType{T, true}) where T<:Number

    out = batched_vec(dP, ψ)

    g.iFT * complex.(cat(out[:,1:2:end], CUDA.zeros(T, p.N_lats, div(p.N_lons,2) + 1 - p.L), dims=2), cat(CUDA.zeros(T,p.N_lats,1), out[:,2:2:end], CUDA.zeros(T, p.N_lats, div(p.N_lons,2) + 1 - p.L), dims=2))
end

SHtoSH_dθ(ψ,m) = transform_SH(SHtoGrid_dθ(ψ,m), m)
SHtoSH_dϕ(ψ,m) = eltype(ψ)(-1) .* SHtoSH_dθ(ψ, m)
SHtoGrid_dϕ(ψ,m) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, m)
SHtoGrid_dϕ(ψ,p,g) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, p, g)

"""
    QG3MM_base(q, p, t)

Base model used in the MM paper, with symmetrization around the equator

dq = - J(ψ,q) - D(ψ,q) + S
"""
function QG3MM_base(q, m, t)
    p, S = m # parameters, forcing vector
    symmetrize_equator!(q, p.p)

    ψ = qprimetoψ(p, q)
    return cat(
    reshape(- J(ψ[1,:,:], q[1,:,:], p) .- D1(ψ, q, p) .+ S[1,:,:], (1, p.p.L, p.p.M)),
    reshape(- J(ψ[2,:,:], q[2,:,:], p) .- D2(ψ, q, p) .+ S[2,:,:], (1, p.p.L, p.p.M)),
    reshape(- J3(ψ[3,:,:], q[3,:,:], p) .- D3(ψ, q, p) .+ S[3,:,:], (1, p.p.L, p.p.M)),
    dims=1)
end

"""
    QG3MM_adv(q, p, t)

Just the Advection

dq = - J(ψ,q)
"""
function QG3MM_adv(q, p, t)
    ψ = qprimetoψ(p, q)
    return permutedims(cat(
    - J(ψ[1,:,:], q[1,:,:], p),
    - J(ψ[2,:,:], q[2,:,:], p),
    - J3(ψ[3,:,:], q[3,:,:], p),
    dims=3),[3,1,2])
end
