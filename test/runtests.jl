using QG3, SciMLBase
using Test

println("testing basic functionality")
include("basic_deriv_and_transform_test.jl")
include("ad_test.jl")
include("basic_test.jl")
include("basic_test_3d.jl")
include("basic_test_gpu.jl")
include("gpu_r2r_test.jl")
include("gpu_cpu_compare.jl")
