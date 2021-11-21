
# q and ψ are always three dimensional, no matter if in real space (level,lon/lat), fourier (level,kx,ky) or spherical (level,il,m)

# right now the equations is solved in real spherical harmonics expansion

# The transform is handled either by naive SH transform or the FastTransforms.jl library and is pre-computed. The FastTransforms.jl is currently invoking aliasing problems and not working for the full model. All SH are handled in the matrix convention that FastTransforms.jl uses: columns by m-value: 0, -1, 1, -2, 2, ..., rows l in ascending order. This is for the naive SH transform definately not the fasted way of storing the coefficients as an additonal allocating reordering needs to be done for every transform. Therefore the coefficient matrix convention is different on GPU, where the columns are ordered 0, 1, 2, .... l_max, 0, -1, -2, -3, ..

# the whole code is written to be differentiable by Zygote. This is why all function are written in a non-mutating way, this is slightly slower on CPU, on GPU some of these functions like batched matrix multicplication should be faster. Even when a GPU is detected, a lot of the pre-computation is done on CPU, the integration however is performed on GPU

# pre-computations are CPU only, the model and integration can be CPU and GPU

# right now it is a bit unconsitant with weather normailization with a==1 \Omega==1 is enforced or not


"""
Convert the streamfunction ψ to (anomlous) potential vorticity q' in spherical harmonics basis

This version is slightly slower than the old one on CPU (as it not aware of the matrix being half full of zeroes), but it is non-mutating which makes it suitable for the automatic differentation.

It replaces the double loop over the coefficient matrix with a batched vector multiply. The advantage other besides it being non-mutating is that it is optimised for GPU, so it might actually be faster on the GPU than doing a manual loop.
"""
function ψtoqprime(p::QG3Model{T}, ψ::AbstractArray{T,3}) where T<:Number
    return cuda_used[] ? reshape(batched_vec(p.Tψq, reshape(ψ,3,:)),3 , p.p.N_lats, p.p.N_lons + 2) : reshape(batched_vec(p.Tψq, reshape(ψ,3,:)),3 , p.p.L, p.p.M)
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
    return cuda_used[] ? reshape(batched_vec(p.Tqψ, reshape(q,3,:)),3 , p.p.N_lats, p.p.N_lons + 2) : reshape(batched_vec(p.Tqψ, reshape(q,3,:)),3 , p.p.L, p.p.M)
end

"""
Compute the Jacobian determinant from ψ and q in μ,λ coordinates, J = ∂ψ/∂x ∂q/∂y - ∂ψ/∂y ∂q/∂x = 1/a^2cosϕ ( - ∂ψ/∂λ ∂q/∂ϕ + ∂ψ/∂ϕ ∂q/∂λ) =  1/a^2 (- ∂ψ/∂λ ∂q/∂μ + ∂ψ/∂μ ∂q/∂λ)

The last term ∂ψ/∂λ accounts for the planetery vorticity, actually it is 2Ω ∂ψ/∂λ, but 2Ω == 1, (write q = q' + 2Ωμ to proof it)

"""
J(ψ::AbstractArray{T,2}, q::AbstractArray{T,2}, m::QG3Model{T}) where T<:Number = transform_SH(SHtoGrid_dμ(ψ, m).*SHtoGrid_dλ(q, m) - (SHtoGrid_dλ(ψ, m).*SHtoGrid_dμ(q, m)), m) - SHtoSH_dλ(ψ, m)

"""
Compute the Jacobian determinant from ψ and q in μ,λ coordinates without the planetary vorticity, as used in computing the eddy/transient forcing
"""
J_F(ψ, q, m::QG3Model{T}) where T<:Number = transform_SH(SHtoGrid_dμ(ψ, m).*SHtoGrid_dλ(q, m) - (SHtoGrid_dλ(ψ, m).*SHtoGrid_dμ(q, m)), m)

"""
For the Jacobian at 850hPa, q = q' + f(1+h/H_0) = q' + f + f*h/H_0, so that the thrid term has to be added.
"""
J3(ψ::AbstractArray{T,2}, q::AbstractArray{T,2}, m::QG3Model{T}) where T<:Number = J(ψ, q + (m.f[3,:,:] - m.f[2,:,:]), m)

"""
Ekman dissipation

 EK = ∇(k∇ψ) = (∇k ∇ψ) + k Δψ
 EK = 1/a^2cos^2ϕ ∂k/∂λ ∂ψ/∂λ + 1/a^2 ∂k/∂ϕ ∂ψ/∂ϕ + k Δψ   (a==1)

 m.∂k∂λ  includes 1/cos^2ϕ
"""
EK(ψ::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = transform_SH(SHtoGrid_dϕ(view(ψ,3,:,:), m) .* m.∂k∂ϕ + SHtoGrid_dλ(view(ψ,3,:,:), m) .* m.∂k∂λ + m.k .* transform_grid(m.Δ .* view(ψ,3,:,:), m), m)

D1(ψ::AbstractArray{T,3}, qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = -TR12(m, ψ) + H(qprime, 1, m)
D2(ψ::AbstractArray{T,3}, qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = TR12(m, ψ) - TR23(m, ψ) + H(qprime, 2, m)
D3(ψ::AbstractArray{T,3}, qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = TR23(m, ψ) + EK(ψ, m) + H(qprime, 3, m)

"""
Temperature relaxation
"""
TR(m::QG3Model{T}, ψ1, ψ2, Ri::T) where T<: Number = m.p.τRi .* Ri .* (ψ1 - ψ2)
TR12(m::QG3Model{T}, ψ::AbstractArray{T,3}) where T<:Number = m.TRcoeffs[1,:,:] .* (ψ[1,:,:] - ψ[2,:,:])
TR23(m::QG3Model{T}, ψ::AbstractArray{T,3}) where T<:Number = m.TRcoeffs[2,:,:] .* (ψ[2,:,:] - ψ[3,:,:])

"""
Horizontal diffusion, q' is anomolous pv (without coriolis) 2D Fields
"""
H(qprime::AbstractArray{T,3}, i::Int, m::QG3Model{T}) where T<: Number = m.p.cH .* (m.∇8 .* qprime[i,:,:])

u(ψ, m) = -m.p.a^(-1) .* SHtoGrid_dϕ(ψ, m)
v(ψ, m) = m.acosϕi .* SHtoGrid_dλ(ψ, m)

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
