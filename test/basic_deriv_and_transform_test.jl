# this is a very basic to test if the transform and derivite work sort of correct, it just checks if it can correctly transform and take derivatives of cosθ
using Test
@testset "Transforms and Derivatives" begin

using QG3, BenchmarkTools, DifferentialEquations, JLD2

# load forcing and model parameters
S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()

qg3p = QG3Model(qg3ppars)

cosθ = similar(qg3p.cosϕ)
msinθ = similar(qg3p.cosϕ)
for i ∈ 1:qg3p.p.N_lats
    cosθ[i,:] .= cos(qg3p.p.colats[i])
    msinθ[i,:] .= -sin(qg3p.p.colats[i])
end
# 2D transforms
cSH_2d = transform_SH(cosθ, qg3p)
@test sum(abs.(cSH_2d) .> 1e-3) == 1
@test abs.(cSH_2d[2,1]) > 0.1

cg = transform_grid(cSH_2d, qg3p)
@test mean(abs.(cg - cosθ) ./ abs.(cg)) < 1e-3

# 2D deriv

# very close to zero
@test abs.(mean(QG3.SHtoGrid_dλ(cSH_2d, qg3p))) < 1e-5

# near constant 1
@test mean(abs.(QG3.SHtoGrid_dμ(cSH_2d, qg3p) .- 1)) < 1e-2

# near -sinθ
@test mean(abs.(QG3.SHtoGrid_dθ(cSH_2d, qg3p) - msinθ)) < 1e-2

# 3D transforms
cosθ = QG3.make3d(cosθ)
msinθ = QG3.make3d(msinθ)

cSH = transform_SH(cosθ, qg3p)

@test mean(abs.(cSH[1,:,:] - cSH_2d)) < 1e-5
@test sum(abs.(cSH) .> 1e-3) == 3
@test abs.(cSH[1,2,1]) > 0.1
@test abs.(cSH[2,2,1]) > 0.1
@test abs.(cSH[3,2,1]) > 0.1

cg = transform_grid(cSH, qg3p)
@test mean(abs.(cg - cosθ) ./ abs.(cg)) < 1e-3


# 3D deriv

# very close to zero
@test abs.(mean(QG3.SHtoGrid_dλ(cSH, qg3p))) < 1e-5

# near constant 1
@test mean(abs.(QG3.SHtoGrid_dμ(cSH, qg3p) .- 1)) < 1e-2

# near -sinθ
@test mean(abs.(QG3.SHtoGrid_dθ(cSH, qg3p) - msinθ)) < 1e-2

end
