# Currently on GPU there exist no r2r FFT plan. This in itself would not be a problem. However, it is a problem for the AD with Zygote which does not work nicely with complex arrays
# Currently on GPU there exist no r2r FFT plan. This in itself would not be a problem. However, it is a problem for the AD with Zygote which does not work nicely with complex arrays

using LinearAlgebra
import Base: *, \, inv, size, ndims, length, show, summary
import LinearAlgebra: mul!
import CUDA.CUFFT
using ChainRulesCore
import ChainRulesCore.rrule

const FORWARD = -1
const BACKWARD = 1

mutable struct cur2rPlan{F,U,T,R,S,V,W} <: AbstractFFTs.Plan{U}
    plan::T
    region::R
    d::S # size in real domain
    n::V  # size in complex domain (internally)  # N/2 + 1, size in complex domain, only used for inverse plan
    ADscale::W # used for reverse mode AD
end

function plan_cur2r(arr::AbstractArray, dims=1)
    plan = CUDA.CUFFT.plan_rfft(arr, dims)
    plan.pinv = CUDA.CUFFT.plan_inv(plan)

    # AD scaling
    halfdim = first(dims)
    d = size(arr, halfdim)
    n = size(plan * arr, halfdim)
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? 1 : 2 for i in 1:n],
        ntuple(i -> i == first(dims) ? n : 1, Val(ndims(arr))),
    )
    scale = [scale; scale] # double cause it's r2r in the format given be to_complex

    return plan_cur2r(plan, dims, d, n, scale)
end

@eval plan_cur2r(plan::AbstractFFTs.Plan{T}, region, d::Integer, n::Integer, scale) where {T} = cur2rPlan{$FORWARD,T,typeof(plan),typeof(region),typeof(d),typeof(n),typeof(scale)}(plan, region, d, n, scale)

# input in arr in real domain
function plan_cuir2r(arr::AbstractArray{T,S}, d::Int, dims=1) where {T,S}
    arr_size = [size(arr)...]
    halfdim = first(dims)
    n = Int(arr_size[halfdim]/2)
    invN = AbstractFFTs.normalization(arr, dims)

    if !(T <: Complex)
      arr_size[idims] = n
      arr_size = Tuple(arr_size)
      arr = CUDA.zeros(Complex{T}, arr_size...)
    end

    twoinvN = 2 * invN
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? invN : twoinvN for i in 1:n],
        ntuple(i -> i == first(dims) ? n : 1, Val(ndims(arr))),
    )
    scale = [scale; scale]

    plan = CUDA.CUFFT.plan_irfft(arr, d, dims)
    plan.pinv = CUDA.CUFFT.plan_inv(plan)
    return plan_cuir2r(plan, dims, d, n, scale)
end

@eval plan_cuir2r(plan::AbstractFFTs.Plan{T}, N::Integer, idim::Integer) where {T} = cur2rPlan{$BACKWARD,T,typeof(plan),typeof(region),typeof(d),typeof(n),typeof(scale)}(plan, region, d, n, scale)

size(p::cur2rPlan, d) = size(p.plan, d)
ndims(p::cur2rPlan) = ndims(p.plan)
length(p::cur2rPlan) = length(p.plan)
show(io::IO, p::cur2rPlan) = print(io, "R2R wrapper of ",p.plan)
summary(p::cur2rPlan) = string("R2R wrapper of ", summary(p.p))

@eval *(p::cur2rPlan{$FORWARD,U,T,R,S,V,W}, x::CuArray) where {U,T,R,S,V,W} = to_real(p.plan * x, p.region)

@eval *(p::cur2rPlan{$BACKWARD,U,T,R,S,V,W}, x::CuArray) where {U,T,R,S,V,W} = p.plan * to_complex(x, p.region, p.n)

@eval \(p::cur2rPlan{$FORWARD,U,T,R,S,V,W}, x::CuArray) where {U,T,R,S,V,W} = p.plan \ to_complex(x, p.region, p.n)

@eval \(p::cur2rPlan{$BACKWARD,U,T,R,S,V,W}, x::CuArray) where {U,T,R,S,V,W} = to_real(p.plan \ x, p.region)

@eval LinearAlgebra.ldiv!(y::CuArray, p::cur2rPlan{U,T,S,$FORWARD}, x::CuArray) where {U,T,S} = LinearAlgebra.ldiv!(y, p.plan, to_complex(x, p.region, p.n))

@eval LinearAlgebra.ldiv!(y::CuArray, p::cur2rPlan{U,T,S,$BACKWARD}, x::CuArray) where {U,T,S} = to_real(LinearAlgebra.ldiv!(y, p.plan, x), p.region)

to_real(input_array::CuArray, region::Integer) where {T} = CUDA.cat(CUDA.real(input_array), CUDA.imag(input_array), dims=region)

function to_complex(input_array::AbstractArray{T,N}, region::Integer, cutoff_ind::Integer) where {T,N}
    Re = selectdim(input_array, region, 1:cutoff_ind)
    Im = selectdim(input_array, region, (cutoff_ind+1):size(input_array,region))

    CUDA.complex.(Re,Im)
end

Zygote.@adjoint function *(P::AbstractFFTs.ScaledPlan, xs)
  return P * xs, function(Δ)
    N = prod(size(xs)[[P.p.region...]])
    return (nothing, N * (P \ Δ))
  end
end
Zygote.@adjoint function \(P::AbstractFFTs.ScaledPlan, xs)
  return P \ xs, function(Δ)
    N = prod(size(Δ)[[P.p.region...]])
    return (nothing, (P * Δ)/N)
  end
end

# adapted from chainrule for rfft
@eval function ChainRulesCore.rrule(::typeof(*),P::cur2rPlan{$FORWARD,U,T,R,S,V,W}, x::AbstractArray{<:Real}) where {U,T,R,S,V,W}
     y = P*x

     scale = P.ADscale
     d = P.d

     project_x = ChainRulesCore.ProjectTo(x)
     function cur2r_pullback(ȳ)
         x̄ = project_x(((P \ ChainRulesCore.unthunk(ȳ)) ./ scale).*d) # instead of brfft, d removes the normalization of the irfft
         return ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), x̄
     end
     return y, cur2r_pullback
 end

# adapted from chainrule for irfft
@eval function ChainRulesCore.rrule(::typeof(*),P::cur2rPlan{$BACKWARD,U,T,R,S,V,W}, x::AbstractArray) where {U,T,R,S,V,W}
    y = P*x

    scale = P.ADscale
    dims = P.region

    project_x = ChainRulesCore.ProjectTo(x)

    function cuir2r_pullback(ȳ)
        x̄ = project_x(scale .* (P \ real(ChainRulesCore.unthunk(ȳ))))
        return ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), x̄
    end
    return y, cuir2r_pullback
end
