# this file is an example how to run from a pre-computed model
import Pkg 
Pkg.activate("examples")

using QG3, BenchmarkTools, OrdinaryDiffEq, JLD2, CUDA

# load forcing and model parameters

# either from files directly
#@load "data/t21-precomputed-S.jld2" S
#@load "data/t21-precomputed-p.jld2" qg3ppars
#@load "data/t21-precomputed-sf.jld2" ψ_0
#@load "data/t21-precomputed-q.jld2" q_0

# or use the function that automatically loads the files that are saved in the repository
S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data(GPU=true)

# pre-computations are partially performed on CPU, so we have to allow scalarindexing
qg3p = CUDA.@allowscalar QG3Model(qg3ppars)
T = eltype(qg3p)
# time step
DT = T(2π/144)
t_end = T(2000.)

# problem definition with standard model from the library and solve
prob = ODEProblem(QG3.QG3MM_gpu, q_0, (T(0.),t_end), (qg3p, S))

vals = SavedValues(T,Vector{T})
sol = @time solve(prob, Tsit5(), dt=DT, callback=SavingCallback(QG3.KineticEnergyCallback(qg3p), vals, saveat=DT))

vals = Matrix(hcat(vals.saveval...))

#=
# PLOT OPtiON
using Plots
plot(vals', title="Kinetic Energy by Layer)
=#
