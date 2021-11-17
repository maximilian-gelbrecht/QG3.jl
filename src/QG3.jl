module QG3

using GSL, CUDA, FFTW, FastGaussQuadrature, Tullio, Flux, StatsBase, LinearAlgebra, FastTransforms, JLD2

import GSL.sf_legendre_deriv_array_e
import GSL.sf_legendre_deriv_alt_array_e
import GSL.sf_legendre_array_index

global const cuda_used = Ref(false)

function __init__() # automatically called at runtime to set cuda_used 
    use_gpu[] = CUDA.functional()
end

using CUDA.CUFFT, CUDAKernels, KernelAbstractions

togpu(x::AbstractArray) = cuda_used[] ? CuArray(x) : x
tocpu(x) = cuda_used ? Array(x) : x

include("basic_types.jl")
include("sph_tools.jl")
include("data_tools.jl")

include("model_precomputations.jl")
include("model.jl")
include("forcing.jl")

export QG3ModelParameters, QG3Model, transform_SH, transform_grid, level_index

export qprimetoψ, ψtoqprime, qtoψ, ψtoq, J, J3, D1, D2, D3

end
