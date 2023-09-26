# this is a very basic to test if the transform and derivite work sort of correct, it just checks if it can correctly transform and take derivatives of cosθ
@testset "Transforms and Derivatives" begin

using QG3, BenchmarkTools, DifferentialEquations, JLD2

# load forcing and model parameters
S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()

qg3p = QG3Model(qg3ppars)
g = qg3p.g

cosθ = zeros(eltype(qg3ppars), qg3ppars.N_lats, qg3ppars.N_lons)
msinθ = zeros(eltype(qg3ppars), qg3ppars.N_lats, qg3ppars.N_lons)
for i ∈ 1:qg3ppars.N_lats
    cosθ[i,:] .= cos(qg3ppars.θ[i])
    msinθ[i,:] .= -sin(qg3ppars.θ[i])
end
# 2D transforms
cSH_2d = transform_SH(cosθ, g)
@test sum(abs.(cSH_2d) .> 1) == 1
@test abs.(cSH_2d[2,1]) > 20.

cg = transform_grid(cSH_2d, g)
@test mean(abs.(cg - cosθ) ./ abs.(cg)) < 1e-3

# back and forward transform
@test isapprox(transform_SH(transform_grid(ψ_0, g),g),ψ_0,rtol=1e-3)

# 2D deriv

# very close to zero
@test abs.(mean(QG3.SHtoGrid_dλ(cSH_2d, g))) < 1e-5

# near constant 1
@test mean(abs.(QG3.SHtoGrid_dμ(cSH_2d, g) .- 1)) < 1e-2

# near -sinθ
@test mean(abs.(QG3.SHtoGrid_dθ(cSH_2d, g) - msinθ)) < 1e-2

# 3D transforms
cosθ = QG3.make3d(cosθ)
cosθ = cat(cosθ, cosθ, cosθ, dims=1)
msinθ = QG3.make3d(msinθ)
msinθ = cat(msinθ, msinθ, msinθ, dims=1)

cSH = transform_SH(cosθ, g)

@test mean(abs.(cSH[1,:,:] - cSH_2d)) < 1e-5
@test sum(abs.(cSH) .> 1.) == 3
@test abs.(cSH[1,2,1]) > 20.
@test abs.(cSH[2,2,1]) > 20.
@test abs.(cSH[3,2,1]) > 20.

cg = transform_grid(cSH, g)
@test mean(abs.(cg - cosθ) ./ abs.(cg)) < 1e-3

# 3D deriv
# very close to zero
@test abs.(mean(QG3.SHtoGrid_dλ(cSH, g))) < 1e-5

# near constant 1
@test mean(abs.(QG3.SHtoGrid_dμ(cSH, g) .- 1)) < 1e-2

# near -sinθ
@test mean(abs.(QG3.SHtoGrid_dθ(cSH, g) - msinθ)) < 1e-2

# batched transforms & derivs 
cosθ = repeat(cosθ, 1,1,1,2)
msinθ = repeat(msinθ, 1,1,1,2)
A = rand(eltype(qg3ppars), size(cosθ)...)

qg3p_4d = QG3Model(qg3ppars; N_batch=2)
g4d = qg3p_4d.g

cSH = transform_SH(cosθ, g4d)

@test transform_SH(cosθ, g) ≈ cSH
@test transform_grid(cSH, g) ≈ transform_grid(cSH, g4d)

ASH = transform_SH(A, g4d)

@test transform_SH(A, g) ≈ ASH
@test transform_grid(ASH, g) ≈ transform_grid(ASH, g4d)

@test QG3.SHtoGrid_dλ(ASH[:,:,:,1],g) ≈ QG3.SHtoGrid_dλ(ASH,g4d)[:,:,:,1]
@test QG3.SHtoGrid_dλ(ASH[:,:,:,2],g) ≈ QG3.SHtoGrid_dλ(ASH,g4d)[:,:,:,2]

@test QG3.SHtoGrid_dμ(ASH[:,:,:,1],g) ≈ QG3.SHtoGrid_dμ(ASH,g4d)[:,:,:,1]
@test QG3.SHtoGrid_dμ(ASH[:,:,:,2],g) ≈ QG3.SHtoGrid_dμ(ASH,g4d)[:,:,:,2]

@test QG3.SHtoGrid_dθ(ASH[:,:,:,1],g) ≈ QG3.SHtoGrid_dθ(ASH,g4d)[:,:,:,1]
@test QG3.SHtoGrid_dθ(ASH[:,:,:,2],g) ≈ QG3.SHtoGrid_dθ(ASH,g4d)[:,:,:,2]

# test Laplacian (there's a bias close to the poles)
msinθ = msinθ[:,:,:,1]
L1 = QG3.SHtoGrid_dθ(transform_SH((-msinθ).*QG3.SHtoGrid_dθ(ψ_0, g4d),g4d),g4d) ./ (-msinθ) + QG3.SHtoGrid_dφ(QG3.SHtoSH_dφ(ψ_0, g4d), g4d) ./ (msinθ .* msinθ)
L2 = transform_grid(QG3.Δ(ψ_0, g4d), g4d)
@test mean(abs.(L1-L2)[:,4:end-4,:]) < 0.05

# batched J test 
ψ_0_4d = repeat(ψ_0,1,1,1,2)
q_0_4d = repeat(q_0,1,1,1,2)
@test QG3.J(ψ_0_4d, q_0_4d, qg3p_4d)[:,:,:,2] ≈ QG3.J(ψ_0, q_0, qg3p)

#  psitoq test 
@test QG3.qprimetoψ(qg3p,QG3.ψtoqprime(qg3p, ψ_0)) ≈ ψ_0

A = QG3.ψtoqprime(qg3p, ψ_0)
A[:,1,1] .= 0

B = (QG3.Δ(ψ_0, qg3p)[1,:,:] - (qg3p.p.R1i) * (ψ_0[1,:,:] - ψ_0[2,:,:]))
B[1,1] = 0 
@test A[1,:,:] ≈ B 

B = (QG3.Δ(ψ_0, qg3p)[2,:,:] + (qg3p.p.R1i) * (ψ_0[1,:,:] - ψ_0[2,:,:])) - (qg3p.p.R2i) * (ψ_0[2,:,:] - ψ_0[3,:,:])
B[1,1] = 0
@test A[2,:,:] ≈ B 

B = (QG3.Δ(ψ_0, qg3p)[3,:,:] + (qg3p.p.R2i) * (ψ_0[2,:,:] - ψ_0[3,:,:]))
B[1,1] = 0
@test A[3,:,:] ≈ B 

# psitoq batched test
@test ψtoqprime(qg3p, ψ_0_4d)[:,:,:,1] ≈ ψtoqprime(qg3p, ψ_0)
@test ψtoqprime(qg3p, ψ_0_4d)[:,:,:,2] ≈ ψtoqprime(qg3p, ψ_0)

@test qprimetoψ(qg3p, q_0_4d)[:,:,:,1] ≈ qprimetoψ(qg3p, q_0)
@test qprimetoψ(qg3p, q_0_4d)[:,:,:,2] ≈ qprimetoψ(qg3p, q_0)

end
