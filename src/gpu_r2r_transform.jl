# Currently on GPU there exist no r2r FFT plan. This in itself would not be a problem. However, it is a problem for the AD with Zygote which does not work nicely with complex arrays
# Currently on GPU there exist no r2r FFT plan. This in itself would not be a problem. However, it is a problem for the AD with Zygote which does not work nicely with complex arrays

using LinearAlgebra
import Base: *, \, inv, size, ndims, length, show, summary
import LinearAlgebra: mul!

const FORWARD = -1
const BACKWARD = 1

mutable struct cur2rPlan{U,T,S,K,R} <: AbstractFFTs.Plan{U}
    plan::T
    region::R
    half_N_p1::S # N/2 + 1
end

function plan_cur2r(arr::CuArray, idims::Integer=1)
    plan = CUDA.CUFFT.plan_rfft(arr, idims)
    plan.inv = CUDA.CUFFT.plan_inv(plan)
    return plan_cur2r(plan, idims)
end

@eval plan_cur2r(plan::AbstractFFTs.Plan{T}, idim::Integer) where {T} = cur2rPlan{T,typeof(plan),Nothing,$FORWARD,typeof(idim)}(plan, idim, nothing)

function plan_cur2r(arr::CuArray, N::Integer, idims::Integer=1)
    plan = CUDA.CUFFT.plan_irfft(arr, N, idims)
    plan.inv = CUDA.CUFFT.plan_inv(plan)
    return plan_icur2r(plan, N, idims)
end

@eval plan_cuir2r(plan::AbstractFFTs.Plan{T}, N::Integer, idim::Integer) where {T} = cur2rPlan{T,typeof(plan),typeof(N),$BACKWARD,typeof(idim)}(plan, idim, (N/2) + 1)

size(p::cur2rPlan, d) = size(p.plan, d)
ndims(p::cur2rPlan) = ndims(p.plan)
length(p::cur2rPlan) = length(p.plan)
show(io::IO, p::cur2rPlan) = print(io, "R2R wrapper of ",p.plan)
summary(p::cur2rPlan) = string("R2R wrapper of ", summary(p.p))

@eval *(p::cur2rPlan{U,T,S,$FORWARD}, x::CuArray) where {U,T,S} = to_real(p.plan * x, p.region)

@eval *(p::cur2rPlan{U,T,S,$BACKWARD}, x::CuArray) where {U,T,S} = p.plan * to_complex(x, p.region, p.half_N_p1)

inv(p::cur2rPlan) = inv(p.plan)

\(p::cur2rPlan, x::CuArray) = p.plan \ x

LinearAlgebra.ldiv!(y::CuArray, p::cur2rPlan, x::CuArray) = LinearAlgebra.ldiv!(y, p.plan, x)

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

Zygote.@adjoint function *(P::cur2rPlan, xs)
  return P * xs, function(Δ)
    N = prod(size(xs)[[P.plan.region...]])
    return (nothing, N * (P \ Δ))
  end
end
Zygote.@adjoint function \(P::cur2rPlan, xs)
  return P \ xs, function(Δ)
    N = prod(size(Δ)[[P.plan.region...]])
    return (nothing, (P * Δ)/N)
  end
end

Zygote.@adjoint function *(P::cur2rPlan{U, <:AbstractFFTs.ScaledPlan, T, K}, xs) where {U,T,K}
  return P * xs, function(Δ)
    N = prod(size(xs)[[P.plan.p.region...]])
    return (nothing, N * (P \ Δ))
  end
end
Zygote.@adjoint function \(P::cur2rPlan{U, <:AbstractFFTs.ScaledPlan, T, K}, xs) where {U,T,K}
  return P \ xs, function(Δ)
    N = prod(size(Δ)[[P.plan.p.region...]])
    return (nothing, (P * Δ)/N)
  end
end
