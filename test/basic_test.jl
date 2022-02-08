# this test just checks if the model successfully compiles and integrates for short times from a pre-computed forcing, Unfortunately this can't test the whole library as this would require a large dataset to compute the forcing from

# this file is an example how to run from a pre-computed model

@testset "Basic 2D/CPU version" begin

using QG3, BenchmarkTools, DifferentialEquations, JLD2

# load forcing and model parameters
@load "../data/t21-precomputed-S.jld2" S
@load "../data/t21-precomputed-p.jld2" qg3ppars
@load "../data/t21-precomputed-sf.jld2" ψ_0
@load "../data/t21-precomputed-q.jld2" q_0

qg3p = QG3Model(qg3ppars)
T =eltype(qg3p)
# time step
DT = T(2π/144)
t_end = T(500.)

# problem definition with standard model from the library and solve
prob = ODEProblem(QG3.QG3MM_base, q_0, (T(0.),t_end), [qg3p, S])
sol = @time solve(prob, AB5(), dt=DT)

@test sol.retcode==:Success
end
