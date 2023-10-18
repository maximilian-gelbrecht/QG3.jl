# this test just checks if the model successfully compiles and integrates for short times from a pre-computed forcing, Unfortunately this can't test the whole library as this would require a large dataset to compute the forcing from

# this file is an example how to run from a pre-computed model
@testset "Basic 3D/GPU version (but on CPU)" begin

using QG3, BenchmarkTools, OrdinaryDiffEq, JLD2

# load forcing and model parameters
S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()

qg3p = QG3Model(qg3ppars)
T = eltype(qg3p)
# time step
DT = T(2π/144)
t_end = T(500.)

# problem definition with standard model from the library and solve
prob = ODEProblem(QG3.QG3MM_gpu, q_0, (T(0.),t_end), (qg3p, S))
sol = @time solve(prob, AB5(), dt=DT)

@test SciMLBase.successful_retcode(sol)
end
