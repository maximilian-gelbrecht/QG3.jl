module QG3

using GSL, CUDA, FFTW, FastGaussQuadrature, Tullio, Flux, StatsBase, LinearAlgebra, FastTransforms, JLD2, Zygote, AbstractFFTs

import GSL.sf_legendre_deriv_array_e
import GSL.sf_legendre_deriv_alt_array_e
import GSL.sf_legendre_array_index

global const cuda_used = Ref(false)

function __init__() # automatically called at runtime to set cuda_used
    cuda_used[] = CUDA.functional()
end

"""
    gpuon()

Manually toggle GPU use on (if available)
"""
function gpuon() # manually toggle GPU use on and off
    cuda_used[] = CUDA.functional()
end

"""
    gpuon()

Manually toggle GPU use off
"""
function gpuoff()
    cuda_used[] = false
end


using CUDA.CUFFT, CUDAKernels, KernelAbstractions

togpu(x::AbstractArray) = cuda_used[] ? CuArray(x) : x
tocpu(x) = cuda_used[] ? Array(x) : x

include("basic_types.jl")
include("sph_tools.jl")
include("data_tools.jl")
include("derivatives.jl")
include("model_precomputations.jl")
include("model.jl")
include("model_gpu.jl")
include("gpu_r2r_transform.jl")
include("forcing.jl")

export QG3ModelParameters, QG3Model, transform_SH, transform_grid, level_index

export qprimetoψ, ψtoqprime, qtoψ, ψtoq, J, D, J3, D1, D2, D3, togpu, tocpu

end
