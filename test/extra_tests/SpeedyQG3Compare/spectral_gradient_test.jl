import Pkg
Pkg.activate(".")

# some preperation borrowed from the speedy gradient test 

using SpeedyWeather, PyPlot

NF = Float64
prog_vars,diag_vars,model_setup = SpeedyWeather.initialize_speedy(NF,trunc=85)
(;spectral_transform,geometry) = model_setup.geospectral
(;radius_earth) = model_setup.parameters

# use only one leapfrog index from vorticitiy
vor = view(prog_vars.vor,:,:,1,:)    

# some large scale initial conditions (zero mean with all l=0 modes being zero)
lmax,mmax = 50,50
vor[2:lmax,2:mmax,:] = randn(Complex{NF},lmax-1,mmax-1,
                                        model_setup.parameters.nlev)
SpeedyWeather.spectral_truncation!(vor,lmax,mmax)   # set upper triangle to 0

vor_start = deepcopy(vor) # copy this for the comparison 

# convert spectral vorticity to spectral stream function and to spectral u,v and transform to u_grid, v_grid
SpeedyWeather.gridded!(diag_vars,prog_vars,model_setup)
(;U_grid, V_grid) = diag_vars.grid_variables    # retrieve u_grid, v_grid from struct
u = zero(vor)
v = zero(vor)

SpeedyWeather.scale_coslat!(V_grid,geometry)

# I copy here for the other two libraries (and transpose because both net latxlon)
u_grid_copy = deepcopy(permutedims(U_grid, (2,1,3)))
v_grid_copy = deepcopy(permutedims(V_grid, (2,1,3)))

SpeedyWeather.spectral!(u,U_grid,spectral_transform)
SpeedyWeather.spectral!(v,V_grid,spectral_transform)

# zonal gradient of v, meridional gradient of u for vorticity
(;coslat_u,coslat_v) = diag_vars.intermediate_variables
SpeedyWeather.gradient_longitude!(coslat_v,v,radius_earth)
SpeedyWeather.gradient_latitude!(coslat_u,u,spectral_transform)


## pyspharm , import and gradient

"""
    pytoc_sph(u::AbstractArray{T,2}, L) where {T}

Converts pyspharm's SPH vector to SpeedyWeather.jl SPH matrix
"""
function pytoc_sph(u::AbstractArray{T,2}, L) where {T}
    out = zeros(T, L, L, size(u,2))
    for lvl ∈ 1:size(u,2)
        u_index = 1
        for i ∈ 1:L 
            out[i:end,i,lvl] = u[u_index:u_index+(L-i),lvl]
            u_index += L - i + 1
        end
    end
    out
end

using PyCall
spharm = pyimport("spharm")
pysh = spharm.Spharmt(length(geometry.lon),length(geometry.lat),gridtype="gaussian", rsphere=1)

py_u = pysh.grdtospec(u_grid_copy, ntrunc=86)
py_v = pysh.grdtospec(v_grid_copy, ntrunc=86)

py_v_dlon ,__ = pysh.getgrad(py_v)
__, py_u_dlat = pysh.getgrad(py_u)

py_vor = py_v_dlon - py_u_dlat

py_vor_spec = pysh.grdtospec(py_vor)
py_vor_spec = pytoc_sph(py_vor_spec, 86)


## QG3.jl

# functions needed to convert QG3.jl SH to SpeedyWeather.jl SH


"""
Shifts the sph coefficients by their l value from a convention that uses the lower triangle to one that uses the upper triangle
"""
function shift_l_ctor(A::AbstractArray{T,3}) where {T}
    B = zeros(T, size(A)...)
    for i ∈ 1:size(A,2)  
        B[1:end-(i-1),i,:] = A[i:end,i,:]
    end 
    B
end 

"""
Shifts the sph coefficients by their l value from a convention that uses the upper triangle to one that uses the lower triangle
"""
function shift_l_rtor(A::AbstractArray{T,3}) where {T}
    B = zeros(T, size(A)...)
    for i ∈ 1:size(A,2)  
        B[:,i:end,i] = A[:,1:end-(i-1),i]
    end 
    B
end 

"""
    ctor_sph(A)

Convert complex sph to real sph (with the proper formula)
"""
function ctor_sph(A)
    CS = sqrt(2)*(-1).^QG3.make3d(QG3.mMatrix(p))

    swap_array_ctor_cpu = [1]
    for i=2:L
        push!(swap_array_ctor_cpu, (i-1)+L)
        push!(swap_array_ctor_cpu, i)
    end 
    swap_array_ctor_cpu

    Re = shift_l_ctor(real.(A)[1:end-1,:,:]) # L=86 not needed
    Im = shift_l_ctor(imag.(A)[1:end-1,:,:])[:,2:end,:] # Im(m=0) isu always zero 
    B = cat(Re,Im,dims=2)
    B = B[:,swap_array_ctor_cpu,:] 
    outp = CS .* permutedims(B,(3,1,2))
    outp[:,:,1] ./= CS[:,:,1]
    outp  # m=0 does not have the sqrt(2)*CS coefficient
end 

"""
    rtoc_sph(A)

Convert real sph to complex sph 
"""
function rtoc_sph(A_in::AbstractArray{T,3}) where {T}
    CS = (-1).^QG3.make3d(QG3.mMatrix(p))/sqrt(2)
    A = CS .* A_in
    A[:,:,1] ./= CS[:,:,1]

    Re = shift_l_rtor(A[:,:,1:2:2*L-1])
    Im = shift_l_rtor(cat(zeros(T,size(A,1),size(A,2)), A[:,:,2:2:2*L-1], dims=3))

    B = complex.(Re,-Im)

    
    permutedims(B, (2,3,1))
end 

using AssociatedLegendrePolynomials
using GSL 
function compare_legendre(L, M, μ::AbstractArray{T,1}) where T
    N_lats = length(μ)
    P_new = zeros(T, N_lats, L, M)

   
    for ilat ∈ 1:N_lats
        for m ∈ -(L-1):(L-1)
            for il ∈ 1:(L - abs(m)) # l = abs(m):l_max
                l = il + abs(m) - 1
                if m<0 # the ass. LP are actually the same for m<0 for our application as only |m| is needed, but I do this here in this way to have everything related to SH in the same matrix format
                    P_new[ilat, il, 2*abs(m)] = λlm(l,abs(m),μ[ilat])
                else
                    P_new[ilat, il, 2*m + 1] = λlm(l,m,μ[ilat])
                end
            end
        end
    end

    __, P = QG3.compute_P(L, M, μ, sh_norm=GSL_SF_LEGENDRE_SPHARM, CSPhase=-1, prefactor=false)
    P_new, P 
end

using QG3, Plots
L = 86
p = QG3ModelParameters(L, geometry.lat, geometry.lon)
g = QG3.grid(p, "gaussian", 8)

Pn, Pold = compare_legendre(p.L, p.M, p.μ) 
# sign is different here for some, still CS phase?

qg3_u = transform_SH(permutedims(u_grid_copy,(3,1,2))[1,:,:], g)
qg3_v = transform_SH(permutedims(v_grid_copy,(3,1,2))[1,:,:], g)

qg3_v_dlon = QG3.SHtoGrid_dλ(qg3_v, g)
qg3_u_dlat = QG3.SHtoGrid_dϕ(qg3_u, g)


# compare QG3.jl to SpeedyWeather.jl 
heatmap(permutedims(u_grid_copy,(3,1,2))[1,:,:])
heatmap(permutedims(v_grid_copy,(3,1,2))[1,:,:])

heatmap(qg3_v_dlon)
heatmap(SpeedyWeather.gridded(coslat_v[:,:,1])')
# different scaling due to R 

# here Speedy returns an additional factor cos(lat) (and also R)
heatmap(qg3_u_dlat)

heatmap(SpeedyWeather.gridded(coslat_u[:,:,1])')

dlat_u = deepcopy(SpeedyWeather.gridded(coslat_u[:,:,1]))
SpeedyWeather.scale_coslat⁻¹!(dlat_u,geometry)
heatmap(dlat_u')

heatmap(qg3_u_dlat - dlat_u')


# compare QG3.jl to pyspharm 
# pyspharm computes the gradient in SPH, this has additional terms due to $$nabla = 1/(r \sin\theta) \partial_\lambda \mathbf{i} + 1/r \partial_\theta \mathbf{j} + \partial_r \mathbf{k}$$ and $$\nabla\psi = \mathbf{v}$


heatmap(py_u_dlat[:,:,1])
heatmap(qg3_u_dlat)

heatmap(py_u_dlat[:,:,1] - qg3_u_dlat)


sinθ = similar(qg3_u_dlat)
for i_l = 1:size(sinθ,1)
    sinθ[i_l,:] .= sin(p.θ[i_l])
end

heatmap(py_v_dlon[:,:,1] .* sinθ)
heatmap(qg3_v_dlon)

heatmap(py_v_dlon[:,:,1] .* sinθ - qg3_v_dlon)
