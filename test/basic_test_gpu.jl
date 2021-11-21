using CUDA
using Test


if CUDA.functional()
    @testset "Basic GPU capability" begin

    using QG3, BenchmarkTools, DifferentialEquations, JLD2

    # load forcing and model parameters
    S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()

    QG3.gpuoff()
    qg3p_cpu = QG3Model(qg3ppars)
    QG3.gpuon()

    S_gpu, qg3ppars_gpu, ψ_0_gpu, q_0_gpu = QG3.reorder_SH_gpu(S, qg3ppars), togpu(qg3ppars), QG3.reorder_SH_gpu(ψ_0, qg3ppars), QG3.reorder_SH_gpu(q_0, qg3ppars)

    qg3p_gpu = QG3Model(qg3ppars_gpu)

    @test mean(abs.(transform_grid(ψ_0_gpu, qg3p_gpu) - transform_grid(ψ_0, qg3p))) < 1e-10

    A = QG3.QG3MM_gpu(q_0_gpu, [qg3p_gpu, S_gpu], 0.)

    B = QG3.QG3MM_base(q_0, [qg3p_cpu, S], 0.)

    mean(abs.(A - B)) < 1e-10
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
