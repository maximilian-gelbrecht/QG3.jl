# this file offers a full matrix / 3d formulation of the model that is geared toward efficient GPU usage, some more general notes of the implementation are given in the model.jl file

"""
    J(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, m::QG3Model{T})

Computes the Jacobian for the QG3 model in 3D Field formulation (e.g. for GPU). Takes as an imput the streamfunction ψ and the potential vorticity q' without the coriolis contribution. Adds the contribution of the Coriolis force and orography at the 850hPa level that is prescribed in the MM QG3 Model. 
"""
J(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, m::QG3Model{T}) where T =  transform_SH(SHtoGrid_dμ(q + m.f_J3, m).*SHtoGrid_dλ(ψ, m) - (SHtoGrid_dλ(q + m.f_J3, m).*SHtoGrid_dμ(ψ, m)), m) - SHtoSH_dλ(ψ, m)

"""
    Jprime(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, g::AbstractGridType{T})

Computes the Jacobian (∂ψ/∂μ ∂q/∂λ - ∂ψ/∂λ ∂q/∂μ). Directly computes the derivatives of the two inputs ψ and q and doesn't account for any additional contribution (e.g. from the Coriolis force).
"""
Jprime(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, g::AbstractGridType{T}) where T =  transform_SH(SHtoGrid_dμ(q, g).*SHtoGrid_dλ(ψ, g) - (SHtoGrid_dλ(q, g).*SHtoGrid_dμ(ψ, g)), g) 
Jprime(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, m::QG3Model{T}) where T = Jprime(ψ,q,m.g)

"""
    H(qprime::AbstractArray{T,3}, m::QG3Model{T})

Computes the horizontal diffusion, q' is anomolous pv (without coriolis) 3D Fields
"""
H(qprime::AbstractArray{T,3}, m::QG3Model{T}) where T = cH∇8(qprime, m)

"""
    TR(ψ::AbstractArray{T,3}, m::QG3Model{T})

Computes the temperature relaxation of all levels
"""
TR(ψ::AbstractArray{T,3}, m::QG3Model{T}) where T = reshape(batched_vec(m.TR_matrix, reshape(ψ,3,:)),3 , m.g.size_SH...) 

"""
    D(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, m::QG3Model{T})

Computes the dissipiation of all levels, 850hPa has additional Ekman dissipation.
"""
D(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, m::QG3Model{T}) where T = add_to_level(TR(ψ, m) + H(q, m), EK(ψ[3,:,:], m), 3)

"""
    add_to_level(ψ::AbstractArray{T,3}, ψ_i::AbstractArray{T,2}, i::Integer)

Differentiable and GPU compatible way of adding only to one of the levels of the model.
"""
function add_to_level(ψ::AbstractArray{T,3}, ψ_i::AbstractArray{T,2}, i::Integer) where T
    ψ[i,:,:] += ψ_i
    ψ
end

Zygote.@adjoint function add_to_level(ψ::AbstractArray{T,3}, ψ_i::AbstractArray{T,2}, i::Integer) where T
    return (add_to_level(ψ, ψ_i::AbstractArray{T,2}, i), Δ->(Δ,Δ,))
end


"""
    QG3MM_gpu(q, p, t)

Base model used in the MM paper, with symmetrization around the equator

dq = - J(ψ,q) - D(ψ,q) + S
"""
function QG3MM_gpu(q, m, t)
    p, S = m # parameters, forcing vector

    ψ = qprimetoψ(p, q)
    return - J(ψ, q, p) - D(ψ, q, p) + S
end

function QG3MM_adv_gpu(q, m, t)
    ψ = qprimetoψ(m, q)
    return - J(ψ, q, m)
end
