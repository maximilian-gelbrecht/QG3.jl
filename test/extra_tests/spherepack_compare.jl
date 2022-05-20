# This test is not included in the full runtests as it needs the pySpherepack installed (which is a bit annoying and slow in CI)
import Pkg
Pkg.activate("test/extra_tests/")

using QG3, PyCall, Plots, StatsBase

# we load the precomputed files and compare transforms on that 
S, qg3ppars, ψ_0, q_0 = QG3.load_precomputed_data()
qg3p = QG3Model(qg3ppars)



spharm = pyimport("spharm")
pysh = spharm.Spharmt(qg3ppars.N_lons,qg3ppars.N_lats,gridtype="gaussian", rsphere=1)


# maps the coefficents (FastTranforms.jl convention)
function sp_to_qg3(x::AbstractArray{Complex{T},1}, p::QG3ModelParameters) where T
    coeff = zeros(T,p.L,p.M)
    CS(m) = (-1)^m
    index = 1

    for m ∈ 0:(p.L-1)

        l_range = 1:(p.L - abs(m))

        if m==0 
            coeff[l_range,2*m+1] = real.(x[index:index + l_range[end] - 1])
        else 
            coeff[l_range,2*m+1] = sqrt(T(2))*CS(m)*real.(x[index:index + l_range[end] - 1])
            coeff[l_range,2*m] = sqrt(T(2))*CS(m)*imag.(x[index:index + l_range[end] - 1])
        end

        index += l_range[end]
    end
    coeff 
end

function sp_to_qg3(x::AbstractArray{Complex{T},2}, p::QG3ModelParameters) where T
    coeff = zeros(T,size(x,2),p.L,p.M)
    for i ∈ 1:size(x,2)
        coeff[i,:,:] = sp_to_qg3(x[:,i], p)
    end
    coeff
end


# test transforms 
A_G = transform_grid(ψ_0, qg3p)

SH_QG3 = transform_SH(A_G, qg3p)
SH_PY = pysh.grdtospec(permutedims(A_G,(2,3,1)),ntrunc=21)

A = SH_QG3
B = sp_to_qg3(SH_PY, qg3p.p)


A[1,:,:]
B[1,:,:]

# transform back 


# using the spherical from pyspharm and then to grid with 

C = transform_grid(SH_QG3, qg3p)
D = pysh.spectogrd(SH_PY) 

D_2 = transform_grid(B, qg3p) * qg3p.p.N_lons

res = permutedims(D,(3,1,2)) ./ D_2
# look into shifting the N_lons to the to SPH, then the to grid is identical at least!
heatmap((C - permutedims(D,(3,1,2)))[1,:,:])
heatmap(A_G[1,:,:] - C[1,:,:])
heatmap(A_G[1,:,:])
heatmap(C[1,:,:])
heatmap([1,:,:])
heatmap(D[:,:,1])


heatmap(C[1,:,:] - D[:,:,1])

heatmap(A_G[1,:,:] - D[:,:,1])


# test SH -> Grid 



# test gradients 
A = transform_grid(ψ_0, qg3p)

A_SH_QG3 = transform_SH(A, qg3p) 
A_SH_PY = pysh.grdtospec(permutedims(A,(2,3,1)), ntrunc=21)

dlat_QG3 = QG3.SHtoGrid_dϕ(A_SH_QG3, qg3p)
dlon_QG3 = QG3.SHtoGrid_dλ(A_SH_QG3, qg3p)

#other lon 
heatmap(transform_grid(QG3.make3d((-QG3.mMatrix(qg3p.p))) .* A_SH_QG3[:,:,qg3p.g.dλ.swap_m_sign_array
],qg3p)[1,:,:])


dlon_PY, dlat_PY = pysh.getgrad(A_SH_PY)

dlon_PY = permutedims(dlon_PY,(3,1,2))
dlat_PY = permutedims(dlat_PY,(3,1,2))

heatmap((dlat_PY - dlat_QG3)[1,:,:])

heatmap((abs.(dlat_PY - dlat_QG3)./abs.(dlat_PY))[1,:,:])

mean(abs.(dlat_PY - dlat_QG3)./abs.(dlat_PY)) < 0.03

heatmap(transform_grid(QG3.make3d((-QG3.mMatrix(qg3p.p))) .* A_SH_QG3[:,:,qg3p.g.dλ.swap_m_sign_array
],qg3p)[1,:,:],clims=(-0.05,0.05))
heatmap(dlon_PY[1,:,:],clims=(-0.05,0.05))
heatmap(dlon_QG3[1,:,:])
heatmap((dlon_PY - dlon_QG3)[1,:,:])



