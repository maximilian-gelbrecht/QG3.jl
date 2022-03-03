# this file offers a full matrix / 3d formulation of the model that is geared toward efficient GPU usage, some more general notes of the implementation are given in the model.jl file

"""
Jacobian in 3D Field formulation (e.g. for GPU), the second term in the q derivatives is the contribution of the f(1+h/H_0) term to the 850hPa level.
"""
J(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number =  transform_SH(SHtoGrid_dμ(ψ, m).*SHtoGrid_dλ(q + m.f_J3, m) - (SHtoGrid_dλ(ψ, m).*SHtoGrid_dμ(q + m.f_J3, m)), m) - SHtoSH_dλ(ψ, m)

"""
Horizontal diffusion, q' is anomolous pv (without coriolis) 3D Fields
"""
H(qprime::AbstractArray{T,3}, m::QG3Model{T}) where T<: Number = (m.cH∇8_3d .* qprime)

"""
Temperature relaxation of all levels
"""
TR(ψ::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = reshape(batched_vec(m.TR_matrix, reshape(ψ,3,:)),3 , m.g.size_SH...) 

"""
Dissipiation of all levels, 850hPa has additional Ekman dissipation
"""
D(ψ::AbstractArray{T,3}, q::AbstractArray{T,3}, m::QG3Model{T}) where T<:Number = add_to_level(TR(ψ, m) + H(q, m), EK(togpu(view(ψ,3,:,:)), m), 3)


"""
Differentiable and GPU compatible way of adding only to one of the levels of the model
"""
function add_to_level(ψ::AbstractArray{T,3}, ψ_i::AbstractArray{T,2}, i::Integer) where T<:Number
    ψ[i,:,:] += ψ_i
    ψ
end

Zygote.@adjoint function add_to_level(ψ::AbstractArray{T,3}, ψ_i::AbstractArray{T,2}, i::Integer) where T<:Number
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
