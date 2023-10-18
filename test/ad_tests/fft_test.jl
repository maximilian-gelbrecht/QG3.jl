cd("test/ad_tests")
import Pkg
Pkg.activate(".")
using FFTW, AbstractFFTs, Zygote, FiniteDifferences, StatsBase, LinearAlgebra
import FFTW.r2r
import ChainRulesCore

# do a 3d plan like we have 



begin 
    Ag = rand(3,32,64)
    Ash = rand(3,22,43)

    Agf = rfft(Ag, 3)
end 
# first directly rfft 

begin 
    y, back = Zygote.pullback(x -> rfft(x, 3), Ag)
    fd_jvp = j′vp(central_fdm(5,1), x -> rfft(x, 3), y, Ag)

    diff = (fd_jvp[1] - back(y)[1]) 
    maximum(abs.(diff)) < 1e-9
end 

# now the plan: 
plan = plan_rfft(Ag, 3)
plan*Ag ≈ y

y, back = Zygote.pullback(x -> plan*x, Ag)
fd_jvp = j′vp(central_fdm(5,1), x -> plan*x, y, Ag)

diff = (fd_jvp[1] - back(y)[1]) 
# thats wrong! the plan ad is incorrect 


# do this for irfft 
y, back = Zygote.pullback(x -> irfft(x, size(Ag,3), 3), Agf)
fd_jvp = j′vp(central_fdm(5,1), x -> irfft(x, size(Ag,3), 3), y, Agf)
diff = (fd_jvp[1] - back(y)[1]) 
maximum(abs.(diff)) < 1e-9



# derive a ad rule for the plan_r2r from the simple forward rule





# directly compare r2r to rfft
hc = FFTW.r2r(Ag, FFTW.R2HC, 3)

hcback = FFTW.r2r(hc, FFTW.HC2R, 3) 
c = rfft(Ag, 3)

real(c) ≈ hc[:,:,1:Int(size(hc,3)/2)+1]
imag(c)[:,:,2:end-1] ≈ hc[:,:,end:-1:Int(size(hc,3)/2)+2] 


# indexing etc is on the FFTW side 
function ChainRulesCore.rrule(::typeof(r2r), x::AbstractArray{<:Real}, FFTW_consts, dims)

    # adapted from AbstractFFTsChainRulesCoreExt rule for rfft/brfft
    y = r2r(x, FFTW.HC2R, dims)
    println("test3")

    # 
    halfdim = first(dims)
    n = size(x, halfdim)
    d = size(y, halfdim)
    i_imag = Int(n/2)+1

    project_x = ChainRulesCore.ProjectTo(x)
    # look up what the do... function does 
    function ir2r_pullback(ȳ)
        x̄_scaled = r2r(ChainRulesCore.unthunk(ȳ), FFTW.R2HC, dims)
        
        x̄ = project_x(map(x̄_scaled, CartesianIndices(x̄_scaled)) do x̄_scaled_j, j
            i = j[halfdim]
            x̄_j = if i == 1 || (i == i_imag && 2 * (i - 1) == d) 
                x̄_scaled_j
            else
                2 * x̄_scaled_j
            end
            return x̄_j
        end)

        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end

    return y, ir2r_pullback
end

# first directly rfft 
y, back = Zygote.pullback(x -> r2r(x, FFTW.R2HC, 3), Ag)
fd_jvp = j′vp(central_fdm(5,1), x -> r2r(x, FFTW.R2HC, 3), y, Ag)

diff = (fd_jvp[1] - back(y)[1]) 
maximum(abs.(diff)) < 1e-9

y, back = Zygote.pullback(x -> r2r(x, FFTW.HC2R, 3), hc)
fd_jvp = j′vp(central_fdm(5,1), x -> r2r(x, FFTW.HC2R, 3), y, hc)

diff = (fd_jvp[1] - back(y)[1]) 
maximum(abs.(diff)) #< 1e-9






include("ad_r2r_plans.jl")

r2r_plan = plan_r2r_AD(Ag, 3)
ir2r_plan = plan_ir2r_AD(Ag, 3)

y, back = Zygote.pullback(x -> r2r_plan*x, Ag)
fd_jvp = j′vp(central_fdm(5,1), x -> r2r_plan*x, y, Ag)
diff_val = (fd_jvp[1] - back(y)[1]) 
maximum(abs.(diff_val)) < 1e-8

y, back = Zygote.pullback(x -> ir2r_plan*x, hc)
fd_jvp = j′vp(central_fdm(5,1), x -> ir2r_plan*x, y, hc)
diff_val = (fd_jvp[1] - back(y)[1]) 
maximum(abs.(diff_val)) < 1e-8



