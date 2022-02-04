
using CUDA, QG3, BenchmarkTools, DifferentialEquations, JLD2,Flux, Zygote

@testset "Basic GPU/CPU comparision" begin

if CUDA.functional()
    # or use the function that automatically loads the files that are saved in the repository
    S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()

    # the precomputed fields are loaded on the CPU and are in the wrong SH coefficient convention
    S_gpu, qg3ppars_gpu, ψ_0_gpu, q_0_gpu = QG3.reorder_SH_gpu(S, qg3ppars), togpu(qg3ppars), QG3.reorder_SH_gpu(ψ_0, qg3ppars), QG3.reorder_SH_gpu(q_0, qg3ppars)

    QG3.gpuoff()
    qg3p = CUDA.@allowscalar QG3Model(qg3ppars);
    QG3.gpuon()
    qg3p_gpu = CUDA.@allowscalar QG3Model(qg3ppars_gpu);

    @test QG3.isongpu(qg3p_gpu)
    @test !(QG3.isongpu(qg3p))


    a = similar(ψ_0)
    a .= 1

    a_gpu = similar(ψ_0_gpu)
    a_gpu .= 1

    function QG3MM_gpu(q)
        ψ = qprimetoψ(qg3p_gpu, q)
        return - a_gpu .* J(ψ, q, qg3p_gpu) - D(ψ, q, qg3p_gpu) + S_gpu
    end

    function QG3MM_cpu(q)
        ψ = qprimetoψ(qg3p, q)
        return - a .* J(ψ, q, qg3p) - D(ψ, q, qg3p) + S
    end

    g2 = @time gradient(Params([a_gpu])) do
        sum(QG3MM_gpu(q_0_gpu))
    end
    A = g2[a_gpu]

    g = @time gradient(Params([a])) do
        sum(QG3MM_cpu(q_0))
    end
    B = g[a];

    B_gpu = QG3.reorder_SH_gpu(B, qg3ppars);

    @test sum(abs.(A - B_gpu)) < 1e-10

    RELTOL = 1e-5
    RELTOL_PREDICT = 1e-3

    DT = (2π/144) / 10 # in MM code: 1/144 * 2π
    t_end = 100.5

    prob_gpu = ODEProblem(QG3.QG3MM_gpu,q_0_gpu,(100.,t_end),[qg3p_gpu, S_gpu])
    sol_gpu = @time solve(prob_gpu, Tsit5(), dt=DT, reltol=RELTOL);

    prob = ODEProblem(QG3.QG3MM_gpu,q_0,(100.,t_end),[qg3p, S])
    sol = @time solve(prob, Tsit5(), dt=DT, reltol=RELTOL);

    diff = abs.(QG3.reorder_SH_gpu(sol(t_end),qg3ppars) - sol_gpu(t_end))./sol_gpu(t_end)
    diff[isnan.(diff)] .= 0;

    @test maximum(diff) < 1e-8

else
    println("CUDA not available. No GPU/CPU comparision tested.")
end
end
