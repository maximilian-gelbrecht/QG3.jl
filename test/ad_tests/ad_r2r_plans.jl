# Currently on GPU there exist no r2r FFT plan. This in itself would not be a problem. However, it is a problem for the AD with Zygote which does not work nicely with complex arrays
# This code also fixes some issues with Zygote and FFT plans for AD

import Base: size, ndims, length, show, summary, *, \, inv
import LinearAlgebra: ldiv!
using CUDA
import CUDA.CUFFT
using ChainRulesCore

abstract type AbstractDifferentiableR2RPlan{F,U} end 

const FORWARD = -1
const BACKWARD = 1
    
struct FFTWR2RPlan{F,U,P,D} <: AbstractDifferentiableR2RPlan{F,U}
    plan::P
    d::Int # size in real domain 
    n::Int # size in frequency domain 
    i_imag::Int # last index of real-valued frequency
    dims::D # dimension(s) along which the FFT is applied
end 
    
@eval function plan_r2r_AD(arr::AbstractArray{T,N}, dims=1) where {T,N} 
    plan = FFTW.plan_r2r(arr, FFTW.R2HC, dims)
    halfdim = first(dims)
    y = plan * arr
    n = size(y, halfdim)
    return FFTWR2RPlan{$FORWARD,T,typeof(plan),typeof(dims)}(plan, size(arr, halfdim), n, Int(n/2)+1, dims) 
end 
    
@eval function plan_ir2r_AD(arr::AbstractArray{T,N}, dims=1) where {T,N}
    plan = FFTW.plan_r2r(arr, FFTW.HC2R, dims)
    halfdim = first(dims)
    y = plan * arr 
    n = size(arr, halfdim)
    return FFTWR2RPlan{$BACKWARD, T, typeof(plan), typeof(dims)}(plan, size(y, halfdim), n, Int(n/2)+1, dims)
end
    
Base.size(p::AbstractDifferentiableR2RPlan, d) = size(p.plan, d)
Base.ndims(p::AbstractDifferentiableR2RPlan) = ndims(p.plan)
Base.length(p::AbstractDifferentiableR2RPlan) = length(p.plan)
Base.show(io::IO, p::AbstractDifferentiableR2RPlan) = print(io, "Differentiable R2R wrapper of ",p.plan)
Base.summary(p::AbstractDifferentiableR2RPlan) = string("Differentiable R2R wrapper of ", summary(p.plan.p))
Base.inv(p::AbstractDifferentiableR2RPlan) = inv(p.plan)

@eval *(p::FFTWR2RPlan{$FORWARD,T}, x::AbstractArray{T}) where T = p.plan * x
@eval *(p::FFTWR2RPlan{$BACKWARD,T}, x::AbstractArray{T}) where T = p.plan * x
@eval \(p::FFTWR2RPlan{$FORWARD,T}, x::AbstractArray{T}) where T  = p.plan \ x
@eval \(p::FFTWR2RPlan{$BACKWARD,T}, x::AbstractArray{T}) where T = p.plan \ x
    
@eval LinearAlgebra.ldiv!(y::AbstractArray{T}, p::FFTWR2RPlan{$FORWARD,T}, x::AbstractArray{T}) where T = LinearAlgebra.ldiv!(y, p.plan, x)
    
@eval LinearAlgebra.ldiv!(y::AbstractArray{T}, p::FFTWR2RPlan{$BACKWARD,T}, x::AbstractArray{T}) where T = LinearAlgebra.ldiv!(y, p.plan, x)
    

struct cur2rPlan{F,U,T,D} <: AbstractDifferentiableR2RPlan{F,U}
    plan::T
    d::Int # size in real domain 
    n::Int # size in frequency domain 
    i_imag::Int # last index of real-valued frequency
    dims::D # dimension(s) along which the FFT is applied
end

@eval function plan_r2r_AD(arr::CuArray{T,N}, dims=1) where {T,N}
    plan = CUDA.CUFFT.plan_rfft(arr, dims)
    plan.pinv = CUDA.CUFFT.plan_inv(plan)

    y = plan * arr
    # AD scaling
    halfdim = first(dims)
    d = size(arr, halfdim)
    i_imag = size(y, halfdim) 
    n = 2 * i_imag 

    return cur2rPlan{$FORWARD,T,typeof(plan),typeof(dims)}(plan, d, n, i_imag, dims)
end

@eval function plan_ir2r_AD(arr::CuArray{T,N}, d::Int, dims=1) where {T,N}
    #here's a complex array needed for making the plan
    plan = CUDA.CUFFT.plan_irfft(arr_complex, d, dims)
    plan.p.pinv = CUDA.CUFFT.plan_inv(plan.p)

    y = plan * arr

    halfdim = first(dims)
    d = 
    n =
    i_imag = # set this correctly to the cutoff index 

    return cur2rPlan{$BACKWARD,T,typeof(plan),typeof(dims)}(plan, d, n, i_imag, dims)
end

@eval *(p::cur2rPlan{$FORWARD,U}, x::CuArray{U}) where U = to_real(p.plan * x, p.dims)
@eval *(p::cur2rPlan{$BACKWARD,U}, x::CuArray{U}) where U = p.plan * to_complex(x, p.dims, p.i_imag) 
@eval \(p::cur2rPlan{$FORWARD,U}, x::CuArray{U}) where U = p.plan \ to_complex(x, p.dims, p.i_imag)
@eval \(p::cur2rPlan{$BACKWARD,U}, x::CuArray{U}) where U = to_real(p.plan \ x, p.dims)
@eval LinearAlgebra.ldiv!(y::CuArray{U}, p::cur2rPlan{$FORWARD,U}, x::CuArray{U}) where U = LinearAlgebra.ldiv!(y, p.plan, to_complex(x, p.region, p.n))
@eval LinearAlgebra.ldiv!(y::CuArray{U}, p::cur2rPlan{$BACKWARD,U}, x::CuArray{U}) where U = to_real(LinearAlgebra.ldiv!(y, p.plan, x), p.region)

to_real(input_array::CuArray{T,N}, region::Integer) where {T,N} = CUDA.cat(CUDA.real(input_array), CUDA.imag(input_array), dims=region)

function to_complex(input_array::AbstractArray{T,N}, region::Integer, cutoff_ind::Integer) where {T,N}
    Re = selectdim(input_array, region, 1:cutoff_ind)
    Im = selectdim(input_array, region, (cutoff_ind+1):size(input_array,region))

    CUDA.complex.(Re,Im)
end

@eval function ChainRulesCore.rrule(::typeof(*), P::AbstractDifferentiableR2RPlan{$FORWARD, T}, x::AbstractArray{T}) where T<:Real
        
    # adapted from AbstractFFTsChainRulesCoreExt rule for rfft
    y = P * x
    
    # compute scaling factors
    halfdim = first(P.dims) # first(P.plan.region)
    d = P.d # d = size(x, halfdim)
    n = P.n # size(y, halfdim)
    i_imag = P.i_imag # Int(n/2)+1
       
    project_x = ChainRulesCore.ProjectTo(x)
    # look up what the do... function does 
    function plan_r2r_pullback(ȳ)
        ybar = ChainRulesCore.unthunk(ȳ)
        # thats applying the function to the input and its indices 
        ybar_scaled = map(ybar, CartesianIndices(ybar)) do ybar_j, j
                i = j[halfdim] # thats the individual index in the dimension we are doing the fft 
                ybar_scaled_j = if i == 1 || (i == i_imag && 2 * (i - 1) == d)
                    ybar_j   # no scaling for first and last frequency of the real part, last one only when input dimension is even (imag part is zero)
                else
                    ybar_j / 2 # all other are scaled 
                end
                return ybar_scaled_j
        end
        x̄ = project_x(inv(P).p * ybar_scaled) # thats an unnormalized inverse transform
    
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
    end
    return y, plan_r2r_pullback
end
    
    
@eval function ChainRulesCore.rrule(::typeof(*), P::AbstractDifferentiableR2RPlan{$BACKWARD, T}, x::AbstractArray{T}) where T<:Real
    # adapted from AbstractFFTsChainRulesCoreExt rule for brfft
    y = P * x
    dims = P.dims #P.plan.region
    # compute scaling factors
    halfdim = first(dims)
    n = P.n #size(x, halfdim)
    d = P.d #size(y, halfdim)
    i_imag = P.i_imag #Int(n/2)+1
        
    project_x = ChainRulesCore.ProjectTo(x)
        
    function plan_ir2r_pullback(ȳ)
        ybar = ChainRulesCore.unthunk(ȳ)
        x̄_scaled = inv(P).p * ybar # R2HC unscaled

        x̄ = project_x(map(x̄_scaled, CartesianIndices(x̄_scaled)) do x̄_scaled_j, j
                i = j[halfdim]
                x̄_j = if i == 1 || (i == i_imag && 2 * (i - 1) == d) 
                    x̄_scaled_j
                else
                    2 * x̄_scaled_j
                end
                return x̄_j
        end)

        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
    end
    return y, plan_ir2r_pullback
end
    