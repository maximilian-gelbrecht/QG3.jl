# here we test that the model is differentiable. It's hard to come up with a proper test, so right now it just looks that Zygote gives no error and the both implementations have the same gradient
@testset "Basic AD capability" begin

using QG3, BenchmarkTools, DifferentialEquations, JLD2, Zygote, Flux

# load forcing and model parameters
S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()

qg3p = QG3Model(qg3ppars)

a = similar(ψ_0)
a .= 1

function QG3MM_gpu(q)
    ψ = qprimetoψ(qg3p, q)
    return - a .* J(ψ, q, qg3p) - D(ψ, q, qg3p) + S
end

function QG3MM_cpu(q)
    ψ = qprimetoψ(qg3p, q)
    return cat(
    reshape(- a[1,:,:] .* J(ψ[1,:,:], q[1,:,:], qg3p) .- D1(ψ, q, qg3p) .+ S[1,:,:], (1,qg3p.p.L, qg3p.p.M)),
    reshape(- a[2,:,:] .* J(ψ[2,:,:], q[2,:,:], qg3p) .- D2(ψ, q, qg3p) .+ S[2,:,:], (1,qg3p.p.L, qg3p.p.M)),
    reshape(- a[3,:,:] .* J3(ψ[3,:,:], q[3,:,:], qg3p) .- D3(ψ, q, qg3p) .+ S[3,:,:],(1,qg3p.p.L, qg3p.p.M)),
    dims=1)
end

mean(abs.(QG3MM_cpu(q_0) - QG3MM_gpu(q_0))) < 1e-8

g2 = gradient(Params([a])) do
    sum(QG3MM_gpu(q_0))
end
A = g2[a]

g = gradient(Params([a])) do
    sum(QG3MM_cpu(q_0))
end
B = g[a]

@test mean(abs.(A - B)) < 1e-10



# here we test one of the extra ad rules that are given so that AD does not use scalar indexing on GPU, we do that be comparing the naive Zygote gradient with the custom gradient that is implemented in the library. On CPU both work, on GPU only the custom gradient will work

swap_array = qg3p.g.dλ.swap_m_sign_array

swap_sign(A::AbstractArray{T,2},swap) where {T} = @inbounds view(A,:,swap)

swap_sign(A::AbstractArray{T,3},swap) where {T} = @inbounds view(A,:,:,swap)

g = gradient(Params([a])) do
    sum(a .* swap_sign(q_0,swap_array))
end
B = g[a]

g2 = gradient(Params([a])) do
    sum(a .* QG3.change_msign(q_0,swap_array))
end
A = g[a]

@test A ≈ B

a = rand(size(a)...)
loss(x) = sum(abs2,QG3.change_msign(a .* QG3.change_msign(q_0,swap_array), swap_array) - x)

loss(q_0)

using Flux
for i=1:5000
    Flux.train!(loss, Flux.params(a), [q_0], ADAM())
end

@test loss(q_0) < 1e-1



end
