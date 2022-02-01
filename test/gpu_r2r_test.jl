
@testset "GPU R2R FFT Wrapper" begin

if CUDA.functional()

    using CUDA.CUFFT
    using Flux
    using QG3

    A = CUDA.rand(100);
    W = CUDA.rand(100);
    V = CUDA.rand(102);

    A2 = CUDA.rand(10,100);
    W2 = CUDA.rand(10,100);

    A3 = CUDA.rand(10,5,100);
    W3 = CUDA.rand(10,5,100);

    fft_plan = plan_cur2r(A, 1)
    ifft_plan = plan_cuir2r(A 1)

    func(x) = ifft_plan*(fft_plan*(W .* x))
    loss(x,y) = sum(abs2,func(x)-y)

    loss(A,A)

    for i=1:2000
        Flux.train!(loss, Flux.params(W), [(A,A)], ADAM())
    end

    @test sum(abs2,W .- 1) < 1e-4

    V = CUDA.rand(102);
    func2(x) = ifft_plan*(W2 .* (fft_plan*x))
    loss2(x,y) = sum(abs2,func2(x)-y)
    loss2(A,A)
    for i=1:2000
        Flux.train!(loss2, Flux.params(V), [(A,A)], ADAM())
    end

    @test loss2(A,A) < 1e-4

    fft_plan = plan_cur2r(A2, 2)
    ifft_plan = plan_cuir2r(fft_plan*A2, 100, 2)

    func(x) = ifft_plan*(fft_plan*(W2 .* x))
    loss(x,y) = sum(abs2,func(x)-y)

    loss(A2,A2)

    for i=1:2000
        Flux.train!(loss, Flux.params(W2), [(A2,A2)], ADAM())
    end

    @test sum(abs2,W2 .- 1) < 1e-3

    fft_plan = plan_cur2r(A3, 3)
    ifft_plan = plan_cuir2r(fft_plan*A3, 100, 3)

    func(x) = ir2r_plan*(r2r_plan*(W3 .* x))
    loss(x,y) = sum(abs2,func(x)-y)

    loss(A3,A3)

    for i=1:1000
        Flux.train!(loss, Flux.params(W3), [(A3,A3)], ADAM())
    end

    @test sum(abs2,W3 .- 1) < 1e-2

else
    println("No CUDA available, test skipped")
end

end
