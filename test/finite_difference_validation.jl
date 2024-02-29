# test if the gradients of the transforms are correct with FiniteDifferences.jl 

using FiniteDifferences
using QG3, Zygote, CUDA, StatsBase

@testset "Transforms AD correctness" begin

    #load forcing and model parameters
    S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()
    
    QG3.gpuoff()
    qg3p = QG3Model(qg3ppars)
    g = qg3p.g

    A = ψ_0
    Ag = Array(transform_grid(ψ_0, g)) # fist I want only FFT test
    
    # first test the r2r plans gradient correctness 
    r2r_plan = QG3.plan_r2r_AD(Ag, 3)
    Agf = r2r_plan * Ag

    ir2r_plan = QG3.plan_ir2r_AD(Agf, size(Ag,3), 3)

    y, back = Zygote.pullback(x -> r2r_plan*x, Ag)
    fd_jvp = j′vp(central_fdm(5,1), x -> r2r_plan*x, y, Ag)
    diff_val = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff_val)) < 1e-4

    yi, backi = Zygote.pullback(x -> ir2r_plan*x, Agf)
    fd_jvpi = j′vp(central_fdm(5,1), x -> ir2r_plan*x, yi, Agf)
    diff_val = (fd_jvpi[1] - backi(yi)[1]) 
    @test maximum(abs.(diff_val)) < 1e-3

    if CUDA.functional() # FiniteDifferences doesn't work on GPU
        Ag_gpu = CUDA.CuArray(Ag)
        Agf_gpu = CUDA.CuArray(Agf)

        r2r_plan_gpu = QG3.plan_r2r_AD(Ag_gpu, 3)
        ir2r_plan_gpu = QG3.plan_ir2r_AD(Agf_gpu, size(Ag_gpu,3), 3)

        cpudiv = (r2r_plan \ Agf);
        gpudiv = (r2r_plan_gpu \ Agf_gpu);
        @test cpudiv ≈ Array(gpudiv)

        gpudiv = Array(ir2r_plan_gpu \ Ag_gpu)
        cpudiv = ir2r_plan \ Ag
        @test gpudiv[:,:,1:33] ≈ cpudiv[:,:,1:33]
        @test gpudiv[:,:,35:end-1] ≈ cpudiv[:,:,end:-1:34]

        y_gpu, back_gpu = Zygote.pullback(x -> r2r_plan_gpu*x, Ag_gpu)
        diff_val = (fd_jvp[1] - Array(back_gpu(y_gpu)[1])) 
        @test maximum(abs.(diff_val)) < 1e-4

        y_gpu, back_gpu = Zygote.pullback(x -> ir2r_plan_gpu*x, Agf_gpu)

        iback_gpu = back_gpu(y_gpu)[1]; 
        diff_val = Array(iback_gpu[:,:,1:33]) - fd_jvpi[1][:,:,1:33]
        @test maximum(abs.(diff_val)) < 1e-4

        diff_val = Array(iback_gpu[:,:,35:end-1]) - fd_jvpi[1][:,:,end:-1:34]
        @test maximum(abs.(diff_val)) < 1e-4
    end 

    
    # test that the AD of the transform are doing what they are supposed to do
    y, back = Zygote.pullback(x -> transform_grid(x, qg3p), A)
    fd_jvp = j′vp(central_fdm(5,1), x -> transform_grid(x, qg3p), y, A)
    diff = (fd_jvp[1] - back(y)[1])
    @test maximum(abs.(diff)) < 1e-4 

    y, back = Zygote.pullback(x -> transform_SH(x, qg3p), Ag)
    fd_jvp = j′vp(central_fdm(5,1), x -> transform_SH(x, qg3p), y, Ag)
    diff = (fd_jvp[1] - back(y)[1])
    @test maximum(abs.(diff)) < 1e-4

    if CUDA.functional() 
        
        QG3.gpuoff()
        qg3p_cpu = QG3Model(qg3ppars)
        QG3.gpuon()

        S_gpu, qg3ppars_gpu, ψ_0_gpu, q_0_gpu = QG3.reorder_SH_gpu(S, qg3ppars), togpu(qg3ppars), QG3.reorder_SH_gpu(ψ_0, qg3ppars), QG3.reorder_SH_gpu(q_0, qg3ppars)

        qg3p_gpu = CUDA.@allowscalar QG3Model(qg3ppars_gpu)
        T = eltype(qg3p_gpu)

        y_cpu, back_cpu = Zygote.pullback(x -> transform_grid(x, qg3p_cpu), AS_cpu)
        y_gpu, back_gpu = Zygote.pullback(x -> transform_grid(x, qg3p_gpu), AS);
        @test maximum(Array(back_gpu(y_gpu)[1])[:,1:22,1:22] - back_cpu(y_cpu)[1][:,1:22,1:2:end]) < 1e-4
        @test maximum(back_cpu(y_cpu)[1][:,1:22,2:2:end] - Array(back_gpu(y_gpu)[1])[:,1:22,35:55]) < 1e-5

        y_cpu, back_cpu = Zygote.pullback(x -> transform_SH(x, qg3p_cpu), AG_cpu)
        y_gpu, back_gpu = Zygote.pullback(x -> transform_SH(x, qg3p_gpu), AG);
        @test Array(back_gpu(y_gpu)[1]) ≈ back_cpu(y_cpu)[1]
    end 

    # test J 
    y, back = Zygote.pullback(x -> QG3.J(q_0, x, qg3p), A)
    fd_jvp = j′vp(central_fdm(11,1), x -> QG3.J(q_0, A, qg3p), y, A)
    diff = (fd_jvp[1] - back(y)[1]) 
    @test mean(abs.(diff)) < 1e-3
    @test maximum(abs.(diff)) < 1e-2

    # test qprimetoψ
    y, back = Zygote.pullback(x -> QG3.qprimetoψ(qg3p, x), q_0)
    fd_jvp = j′vp(central_fdm(5,1), x -> QG3.qprimetoψ(qg3p, x), y, q_0)
    diff = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff)) < 1e-4

    # test H    
    y, back = Zygote.pullback(x -> QG3.H(x, qg3p), A)
    fd_jvp = j′vp(central_fdm(5,1), x -> QG3.H(x, qg3p), y, A)
    diff = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff)) < 1e-8

    # test TR 
    y, back = Zygote.pullback(x -> QG3.TR(x, qg3p), A)
    fd_jvp = j′vp(central_fdm(5,1), x -> QG3.TR(x, qg3p), y, A)
    diff = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff)) < 1e-6

    # test EK 
    y, back = Zygote.pullback(x -> QG3.EK(x, qg3p), A[3,:,:])
    fd_jvp = j′vp(central_fdm(5,1), x -> QG3.EK(x, qg3p), y, A[3,:,:])
    diff = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff)) < 1e-6

    # test D 
    y, back = Zygote.pullback(x -> QG3.D(x, q_0, qg3p), A)
    fd_jvp = j′vp(central_fdm(8,1), x -> QG3.D(x, q_0, qg3p), y, A)
    diff = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff)) < 1e-6
 
    # test complete RHS    
    y, back = Zygote.pullback(x -> QG3.QG3MM_gpu(x, [qg3p, S], 0), A)
    fd_jvp = j′vp(central_fdm(15,1), x -> QG3.QG3MM_gpu(x, [qg3p, S], 0), y, A)
    diff = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff)) < 1e-6
end

