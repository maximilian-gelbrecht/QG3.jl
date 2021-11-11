module QG3

using GSL, CUDA, FFTW, FastGaussQuadrature, Tullio, Flux, StatsBase, LinearAlgebra, FastTransforms

import GSL.sf_legendre_deriv_array_e
import GSL.sf_legendre_deriv_alt_array_e
import GSL.sf_legendre_array_index

global const cuda_used = CUDA.functional()
togpu(x) = cuda_used ? CuArray(x) : x
tocpu(x) = cuda_used ? Array(x) : x

if cuda_used
    using CUDA.CUFFT, CUDAKernels, KernelAbstractions
end

include("basic_types.jl")
include("sph_tools.jl")
include("data_tools.jl")

include("model_precomputations.jl")
include("model.jl")
include("forcing.jl")

export QG3ModelParameters, QG3Model, transform_SH, transform_grid, level_index, togpu, tocpu

export qprimetoψ, ψtoqprime, qtoψ, ψtoq, J, J3, D1, D2, D3

end
