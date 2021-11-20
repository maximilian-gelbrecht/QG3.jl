# here we test that the model is differentiable. It's hard to come up with a proper test, so right now it just looks that Zygote gives no error and the both implementations have the same gradient
@testset "Basic AD capability" begin

using QG3, BenchmarkTools, DifferentialEquations, JLD2, Zygote

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

end
