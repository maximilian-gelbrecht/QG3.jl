
@testset "GPU R2R FFT Wrapper" begin

if CUDA.functional()

    using CUDA.CUFFT
    using Flux
    using QG3

    A = CUDA.rand(100);
    W = CUDA.rand(100);

    A2 = CUDA.rand(10,100);
    W2 = CUDA.rand(10,100);

    A3 = CUDA.rand(10,5,100);
    W3 = CUDA.rand(10,5,100);

    _fft_plan = CUDA.CUFFT.plan_rfft(A, 1)
    _ifft_plan = CUDA.CUFFT.plan_irfft(_fft_plan*A, size(A,1), 1)

    fft_plan = plan_cur2r(_fft_plan, size(A,1), 1)
    ifft_plan = plan_cuir2r(_ifft_plan, size(A,1), 1)

    func(x) = ifft_plan*(fft_plan*(W .* x))
    loss(x,y) = sum(abs2,func(x)-y)

    loss(A,A)

    for i=1:2000
        Flux.train!(loss, Flux.params(W), [(A,A)], ADAM())
    end

    @test sum(abs2,W .- 1) < 1e-4

    _fft_plan = CUDA.CUFFT.plan_rfft(A2, 2)
    _ifft_plan = CUDA.CUFFT.plan_irfft((_fft_plan*A2), size(A2,2), 2)

    fft_plan = plan_cur2r(_fft_plan, size(A2,2), 2)
    ifft_plan = plan_cuir2r(_ifft_plan, size(A2,2), 2)

    func(x) = ifft_plan*(fft_plan*(W2 .* x))
    loss(x,y) = sum(abs2,func(x)-y)

    loss(A2,A2)

    for i=1:2000
        Flux.train!(loss, Flux.params(W2), [(A2,A2)], ADAM())
    end

    @test sum(abs2,W2 .- 1) < 1e-3


    _fft_plan = CUDA.CUFFT.plan_rfft(A3, 3)
    _ifft_plan = CUDA.CUFFT.plan_irfft(_fft_plan*A3, size(A2,3), 3)

    fft_plan = plan_cur2r(_fft_plan, size(A3,3), 3)
    ifft_plan = plan_cuir2r(_ifft_plan, size(A3,3), 3)

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
