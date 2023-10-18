# test if the gradients of the transforms are correct with FiniteDifferences.jl 

using FiniteDifferences
using QG3, Zygote, CUDA

@testset "Transforms AD correctness" begin

    #load forcing and model parameters
    S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()

    qg3p = QG3Model(qg3ppars)
    g = qg3p.g

    A = ψ_0
    Ag = transform_grid(ψ_0, g) # fist I want only FFT test
    
    # first test the r2r plans gradient correctness 
    r2r_plan = QG3.plan_r2r_AD(Ag, 3)
    Agf = r2r_plan * Ag

    ir2r_plan = QG3.plan_ir2r_AD(Agf, 3)

    y, back = Zygote.pullback(x -> r2r_plan*x, Ag)
    fd_jvp = j′vp(central_fdm(5,1), x -> r2r_plan*x, y, Ag)
    diff_val = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff_val)) < 1e-4

    y, back = Zygote.pullback(x -> ir2r_plan*x, Agf)
    fd_jvp = j′vp(central_fdm(5,1), x -> ir2r_plan*x, y, Agf)
    diff_val = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff_val)) < 1e-3

    if CUDA.functional()

        # test also the correctness of the GPU version

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

        # test also the correctness of the GPU version


    end 



end

