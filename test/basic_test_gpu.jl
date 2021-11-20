using CUDA

if CUDA.functional()

    using QG3, BenchmarkTools, DifferentialEquations, JLD2

    # load forcing and model parameters
    S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()
    S, qg3ppars, ψ_0, q_0 = togpu(S), togpu(qg3ppars), QG3.reorder_SH_gpu(ψ_0), QG3.reorder_SH_gpu(q_0)

    qg3p = QG3Model(qg3ppars)

    # time step
    DT = 2π/144
    t_end = 500.

    # problem definition with standard model from the library and solve
    prob = ODEProblem(QG3.QG3MM_gpu, q_0, (0.,t_end), [qg3p, S])
    sol = @time solve(prob, AB5(), dt=DT)

    if sol.retcode==:Success
            return true
    else
            return false;
    end




else
    println("No CUDA available, test skipped")
end
