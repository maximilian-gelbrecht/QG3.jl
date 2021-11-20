# this file contains function that compute Spherical Harmonics
# Right now there are three different variants possible
# a) A self implemented Gaussian Grid based version
# b) Fast Transforms.jl for regular grids
#
#




# shared functions

"""

Pre-compute ass. Legendre Polynomials and dP/dx (derivative of ass. Legendre Polynomial) at the grid points and also the remainder of the Spherical Harmonics at the grid points using GSL

m values are stored 0,-1,1,-2,2,-3,3,... (on CPU)
m values are stored 0,1,2,3,4,5,6,7, ...l_max, 0 (nothing),-1, -2, -3, (on GPU)  (the second 0 is the Imanigary part / sin part of the fourier transform which is always identical to zero, it is kept here to have equal matrix sizes)

# so far only |m| is used, as I assume real SPH.


"""
function compute_P(L::Integer, M::Integer, μ::AbstractArray{T,1}; sh_norm=GSL_SF_LEGENDRE_FULL, CSPhase::Integer=-1) where T<:Number

    N_lats = length(μ)
    P = zeros(T, N_lats, L, M)
    dPμdμ = zeros(T, N_lats, L, M)
    dPcosθdθ = zeros(T, N_lats, L, M)

    gsl_legendre_index(l,m) = m > l ? error("m > l, not defined") : sf_legendre_array_index(l,m)+1 # +1 because of c indexing vs julia indexing

    for ilat ∈ 1:N_lats
        temp = sf_legendre_deriv_array_e(sh_norm, L - 1, μ[ilat], CSPhase) # cordon sherley factor? -1 or 1?
        temp_alt = sf_legendre_deriv_alt_array_e(sh_norm, L - 1, μ[ilat], CSPhase) # cordon sherley factor? -1 or 1?

        for m ∈ -(L-1):(L-1)
            for il ∈ 1:(L - abs(m)) # l = abs(m):l_max
                l = il + abs(m) - 1
                if m<0 # the ass. LP are actually the same for m<0 for our application as only |m| is needed, but I do this here in this way to have everything related to SH in the same matrix format
                    P[ilat, il, 2*abs(m)] = temp[1][gsl_legendre_index(l,abs(m))]
                    dPμdμ[ilat, il, 2*abs(m)] = temp[2][gsl_legendre_index(l,abs(m))]
                    dPcosθdθ[ilat, il, 2*abs(m)] =  temp_alt[2][gsl_legendre_index(l,abs(m))]
                else
                    P[ilat, il, 2*m+1] = temp[1][gsl_legendre_index(l,m)]
                    dPμdμ[ilat, il, 2*m+1] = temp[2][gsl_legendre_index(l,m)]
                    dPcosθdθ[ilat, il, 2*m+1] = temp_alt[2][gsl_legendre_index(l,m)]
                end
            end
        end
    end
    return dPμdμ, dPcosθdθ, P
end
compute_P(p::QG3ModelParameters; kwargs...) = compute_P(p.L, p.M, p.μ; kwargs...)

"""
Reorders the SH coefficient so that computations on GPU are more efficient, it also includes the truncation as the SH array has size N_lat x N_lon, however all entries outside of L x M are zeroes.

Original order (FastTransforms.jl) columns by m : 0, -1, 1, -2, -2, ....

New order columns by m: 0, 1, 2, ... l_max, 0 (nothing), -1, -2, ..

2d input assumes L x M matrix, (all fields in SH), also enlarges the matrix to include truncation (additional elements are zero) to N_lat x N_lons
3d input assumes N_lat (or 3) x L x M matrix (e.g. precomputed legendre polynomials), also enlarges the matrix to include truncation (additional elements are zero) to N_lat (or 3) x N_lats x N_lons
"""
function reorder_SH_gpu(A::AbstractArray{T,2}, p::QG3ModelParameters{T}) where T<:Number
    reindex = [1:2:p.N_lons; 2:2:p.N_lons]
    out = zeros(T, p.N_lats, p.N_lons)
    out[1:p.L, 1:p.M] = A
    return togpu(A[:,reindex])
end

function reorder_SH_gpu(A::AbstractArray{T,3}, p::QG3ModelParameters{T}) where T<:Number
    reindex = [1:2:p.N_lons; 2:2:p.N_lons]

    out = zeros(T, size(A, 1), p.N_lats, p.N_lons)
    out[:, 1:p.L, 1:p.M] = A
    return togpu(A[:,:,reindex])
end

function get_uppertriangle_sum(A)
    cumsum = 0
    for i=1:size(A,1)
        cumsum += sum(abs,A[i,1:end-2*(i-1)])
    end
    cumsum
end

function get_lowertriangle_sum(A)
    cumsum = 0.
    for i=2:size(A,1)
        cumsum += sum(abs,A[i,end-2*(i-1):end])
    end
    cumsum
end

"""
    change_msign(A)

Change the sign of the m in SH (FastTranforms.jl convention of storing them). This version swaps the columns inplace
"""
function change_msign!(A)
    for i=1:(size(A,1)-1)
        A[:,2*i], A[:,2*i+1] = A[:,2*i+1], A[:,2*i]
    end
    A
end

"""
    change_msign(A)

Change the sign of the m in SH (FastTranforms.jl convention of storing them). This version returns a view

there is currently a bug or at least missing feature in Zygote, the AD library, that stops views from always working flawlessly when a view is mixed with prior indexing of an array. We need a view for the derivative after φ to change the sign of m, so here is a differentiable variant of the SHtoSH_dφ function for the 2d field
"""
change_msign(A::AbstractArray{T,2}, swap_array) where T<:Number = view(A,:,swap_array)

function change_msign(A::AbstractArray{T,3}, i::Int) where T<:Number
    arr = [1;vcat([[2*i+1,2*i] for i=1:size(A,2)-1]...)]
    _change_msign(A, i, arr)
end

_change_msign(A::AbstractArray{T,3}, i, arr) where T<:Number = @inbounds view(A,i,:,arr)

# 3d field version
change_msign(A::AbstractArray{T,3}, swap_array) where T<:Number = @inbounds view(A,:,:,swap_array)



"""
Return l-Matrix of SH coefficients in convention of FastTransforms.jl
"""
function lMatrix(L, M)
    l = zeros(Int, L, M)

    for m ∈ -(L-1):(L-1)
        im = m<0 ? 2*abs(m) : 2*m+1
        l[1:L-abs(m),im] = abs(m):(L-1)
    end

    return l
end
lMatrix(p::QG3ModelParameters) = lMatrix(p.L, p.M)
lMatrix(p::QG3Model) = lMatrix(p.p)

"""
Pre-compute a matrix with (m) values of the SH matrix format of FastTransforms.jl, used for zonal derivative
"""
function compute_mmMatrix(L::Integer, M::Integer) where T<:Number
    mmMat = zeros(L,M)
    for m ∈ -(L-1):(L-1)
        for il ∈ 1:(L - abs(m))
            if m<0
                mmMat[il, 2*abs(m)] = m
            else
                mmMat[il, 2*m+1] = m
            end
        end
    end
    mmMat
end
compute_mmMatrix(p::QG3ModelParameters{T}) where T<:Number = T.(compute_mmMatrix(p.L,p.M))
compute_mmMatrix(p::QG3Model) = compute_mmMatrix(p.p)



# test this
# Y_lm(π - θ, ϕ) = (-1)^(l+m) Y_lm(θ, ϕ)
# thus even l+m symmetric, odd l+m anti-symmetric
function symmetrize_equator!(ψ::AbstractArray{T,2}, p::QG3ModelParameters{T}) where T<:Number

    m = compute_mmMatrix(p)
    l = lMatrix(p)

    ψ[mod.(l + abs.(m), 2) .== 0] .= T(0)
end

function symmetrize_equator!(ψ::AbstractArray{T,3}, p::QG3ModelParameters{T}) where T<:Number
    for i=1:3
        symmetrize_equator!(ψ[i,:,:], p)
    end
end

# a) GaussianGrid

"""
Pre-computes gaussian weights for Legendre Transform, also checks if we really have the correct gaussian latitudes
"""
function compute_GaussWeights(p::QG3ModelParameters{T}, reltol=1e-2) where T<:Number
     nodes, weights = gausslegendre(p.N_lats)
     N_lats2 = Int(p.N_lats/2)
     μ = tocpu(p.μ)

     # get the order right, nodes is counting up
     nodes = μ[1] > 0 ? reverse(nodes) : nodes
     weights =  μ[1] > 0 ? reverse(weights) : weights

     # check if the Gaussian latitudes are correct
     check = (nodes[1:N_lats2] .- (reltol .* nodes[1:N_lats2])) .<= μ[1:N_lats2] .<= (nodes[1:N_lats2] .+ (reltol .* nodes[1:N_lats2]))

     if sum(check)!=N_lats2
         error("Gaussian Latitudes not set currently")
     end

     T.(weights)
end

function compute_LegendreGauss(p::QG3ModelParameters{T}, P::AbstractArray{T,3},w::AbstractArray{T,1}) where T<:Number
    # P in format lat x L x M
    for i=1:p.N_lats
        P[i,:,:] *= w[i]
    end
    P
end

function compute_LegendreGauss(p::QG3ModelParameters{T}, P::AbstractArray{T,3}; reltol::Number=1e-2) where T<:Number
    w = compute_GaussWeights(p, reltol)
    return compute_LegendreGauss(p, P, w)
end


"""
manually implemented transform from spherical harmonics to grid. This works for all grid types, the grid type is indirectly specified through the pre-computed ass. legendre polynomials

CPU variant, for 2D Field
"""
function transformSHtoGGrid(A::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T, false}) where T<:Number

    out = batched_vec(g.P, A)

    # pad with zeros and adjust to indexing of FFTW
    g.iFT * cat(out[:,1:2:end], zeros(T, p.N_lats, p.N_lons - p.M), out[:,end-1:-2:2], dims=2)
end

"""
manually implemented transform from spherical harmonics to grid. This works for all grid types, the grid type is indirectly specified through the pre-computed ass. legendre polynomials

CPU variant, for 3D Array
"""
function transformSHtoGGrid(A::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::GaussianGrid{T, false}) where T<:Number

    @tullio out[lvl, ilat, im] := g.P[ilat, il, im] * A[lvl, il, im]

    # pad with zeros and adjust to indexing of FFTW
    g.iFT_3d * cat(out[:,:,1:2:end], zeros(T, 3, p.N_lats, p.N_lons - p.M), out[:,:,end-1:-2:2], dims=3)
end

# GPU/CUDA variant
function transformSHtoGGrid(A::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T, true}) where T<:Number
    out = batched_vec(g.P,A)

    Re = @view out[:,1:p.L]
    Im = @view out[:,p.L+1:end]

    g.iFT * complex.(Re, Im)
end

# GPU/CUDA variant for 3d field
function transformSHtoGGrid(A::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::GaussianGrid{T, true}) where T<:Number
    @tullio out[ilvl, ilat, im] := g.P[ilat, il, im] * A[ilvl, il, im]

    Re = @view out[:,:,1:p.L]
    Im = @view out[:,:,p.L+1:end]

    g.iFT_3d * complex.(Re, Im)
end

"""
manually implemented transfrom from grid to spherical space, so far only for gaussian grid

CPU variant
"""
function transformGGridtoSH(A::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T, false}) where T<:Number

    FTA = (g.FT * A)[:,g.truncate_array] ./ p.N_lons # has to be normalized as this is not done by FFTW

    @tullio out[il,im] := g.Pw[i,il,im] * FTA[i,im]
end

"""
manually implemented transfrom from grid to spherical space, so far only for gaussian grid

CPU variant, 3D vectorized variant
"""
function transformGGridtoSH(A::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::GaussianGrid{T, false}) where T<:Number

    FTA = (g.FT_3d * A)[:,:,g.truncate_array] ./ p.N_lons # has to be normalized as this is not done by FFTW

    @tullio out[ilvl,il,im] := g.Pw[ilat,il,im] * FTA[ilvl,ilat,im]
end


"""
manually implemented transfrom from grid to spherical space, so far only for gaussian grid

GPU variant, r2c fft instead of r2r fft only for the full 3d field
"""
function transformGGridtoSH(A::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T, true}) where T<:Number

    FTA = g.FT * A

    # deal with the complex array, turn it into half complex format
    FTA_HC = reinterpret(T, FTA)
    Re_FTA = @view FTA_HC[1:2:(end-1), 1:p.L]
    Im_FTA = @view FTA_HC[2:2:end, 2:p.L]

    HCr = cat(Re_FTA, Im_FTA, dims=2)

    # truncation is performed in this step as Pw has 0s where the expansion is truncated
    @tullio out[il,im] := g.Pw[i,il,im] * HCr[i,im]
end

function transformGGridtoSH(A::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::GaussianGrid{T, true}) where T<:Number

    FTA = g.FT_3d * A

    # deal with the complex array, turn it into half complex format
    FTA_HC = reinterpret(T, FTA)
    Re_FTA = @view FTA_HC[1:2:(end-1), 1:p.L]
    Im_FTA = @view FTA_HC[2:2:end, 2:p.L]

    HCr = cat(Re_FTA, Im_FTA, dims=2)

    # truncation is performed in this step as Pw has 0s where the expansion is truncated
    @tullio out[ilvl,il,im] := g.Pw[ilat,il,im] * HCr[ilvl,ilat,im]
end


#b) FastTransform.jl

transform_SH_FT(data::AbstractArray{T,2}, planSH::FastTransforms.FTPlan, planFT::FastTransforms.FTPlan, P; setzeros=false) where T<:Number = setzeros ? sph_zero_spurious_modes(togpu(planSH\(planFT*data)), P) : planSH\(planFT*data)

"""
Sets the unphysical numbers zero (usually only needed for FastTransforms.jl)

"""
function sph_zero_spurious_modes!(A::AbstractMatrix)
    M, N = size(A)
    n = N÷2
    @inbounds for j = 1:n-1
        @inbounds for i = M-j+1:M
            A[i,2j] = 0
            A[i,2j+1] = 0
        end
    end
    @inbounds for i = M-n+1:M
      A[i,2n] = 0
      2n < N && (A[i,2n+1] = 0)
    end
    A
end

sph_zero_spurious_modes(A::AbstractMatrix, P) = P.*A

function prepare_sph_zero_spurious_modes(p::QG3ModelParameters)
    M, N = p.N_lats, p.N_lons
    n = N÷2
    P = ones(Bool,M,N)
    @inbounds for j = 1:n-1
        @inbounds for i = M-j+1:M
            P[i,2j] = 0
            P[i,2j+1] = 0
        end
    end
    @inbounds for i = M-n+1:M
      P[i,2n] = 0
      2n < N && (P[i,2n+1] = 0)
    end
    P
end

function sph_zero_spurious_modes!(A::AbstractArray{T,3}) where T<:Number
    L = size(A,1)
    for lvl ∈ 1:L
        A[lvl,:,:] = sph_zero_spurious_modes!(A[lvl,:,:])
    end
    return A
end
function sph_zero_spurious_modes!(A::AbstractArray{T,4}) where T<:Number
    Nt = size(A,4)
    for it ∈ 1:Nt
        A[:,:,:,it] = sph_zero_spurious_modes!(A[:,:,:,it])
    end
    return A
end

truncate(A::AbstractArray{T,2}, p::QG3ModelParameters{T}) where T<:Number = A[1:p.L,1:p.M]
truncate(A::AbstractArray{T,3}, p::QG3ModelParameters{T}) where T<:Number = A[:,1:p.L,1:p.M]
truncate(A::AbstractArray{T,4}, p::QG3ModelParameters{T}) where T<:Number = A[:,1:p.L,1:p.M,:]
truncate(A, m::QG3Model) = truncate(A, m.p)

function revert_truncate(A::AbstractArray{T,2}, p::QG3ModelParameters{T}) where T<:Number
    if p.L != p.N_lats
        B = zeros(T,p.N_lats,p.N_lons)
        B[1:p.L, 1:p.M] = A
        return B
    else
        return A
    end
end

### all transforms

"""
transform_SH(data, m::QG3Model, varname::String="ψ", setzeros=true)

Transforms the data to spherical harmonics.

if setzeros == true, the "unphysical" lower triangle is directly set zero when using FastTransforms.jl
"""
transform_SH(data::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T}; kwargs...) where T<:Number = transformGGridtoSH(data, p, g)

transform_SH(data::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::RegularGrid{T}; kwargs...) where T<:Number = truncate(transform_SH_FT(data, g.SH, g.FT, g.P_spurious_modes; setzeros=g.set_spurious_zero), p)


transform_SH(data::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::GaussianGrid; kwargs...) where T<:Number = transformGGridtoSH(data, p, g)

function transform_SH(data::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::AbstractGridType{T,false}; kwargs...) where T<:Number
    if size(data,1)!=3
        @error("First dimension is not three")
    end
    data_sh = zeros(T,3,p.L,p.M)
    for i ∈ 1:3
        data_sh[i,:,:] = transform_SH(data[i,:,:], p, g; kwargs...)
    end
    return data_sh
end

function transform_SH(data::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::AbstractGridType{T,true}; kwargs...) where T<:Number
    if size(data,1)!=3
        @error("First dimension is not three")
    end
    data_sh = CUDA.zeros(T,3,p.L,p.M)
    for i ∈ 1:3
        data_sh[i,:,:] = transform_SH(data[i,:,:], p, g; kwargs...)
    end
    return data_sh
end

function transform_SH(data::AbstractArray{T,4}, p::QG3ModelParameters{T}, g::AbstractGridType{T,false}; kwargs...) where T<:Number
    data_sh = zeros(T, 3, p.L, p.M,size(data,4))
    for it ∈ 1:size(data,4)
        data_sh[:,:,:,it] = transform_SH(data[:,:,:,it], p, g; kwargs...)
    end
    return data_sh
end

function transform_SH(data::AbstractArray{T,4}, p::QG3ModelParameters{T}, g::AbstractGridType{T,true}; kwargs...) where T<:Number
    data_sh = CUDA.zeros(T, 3, p.L, p.M,size(data,4))
    for it ∈ 1:size(data,4)
        data_sh[:,:,:,it] = transform_SH(data[:,:,:,it], p, g; kwargs...)
    end
    return data_sh
end
transform_SH(data::AbstractArray, m::QG3Model; kwargs...) = transform_SH(data, m.p, m.g; kwargs...)




"""
transform_grid(data, m::QG3Model)

Transforms the data to real space

If varname ∈ ["pv","q","vorticity"] the [1,1] element is set to zero first, as this element is set to the streamfunction-[1,1] usually in the ψtoq routines.

"""
function transform_grid(data::AbstractArray{T,2}, m::QG3Model; varname::String="ψ", kwargs...) where T<:Number
     if varname ∈ ["pv","q","vorticity"]
         data_out = copy(data)
         data_out[1,1] = T(0)
         return transform_grid(data_out, m.p, m.g; kwargs...)
     else
         return transform_grid(data, m.p, m.g; kwargs...)
     end
 end

transform_grid(data::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T}; kwargs...) where T<:Number = transformSHtoGGrid(data, p, g)

transform_grid(data::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::RegularGrid{T}; kwargs...) where T<:Number = togpu(g.FTinv*(g.SH*revert_truncate(data, p)))

transform_grid(data::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::GaussianGrid; varname::String="ψ") where T<: Number = transformSHtoGGrid(data, p, g)

function transform_grid(data::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::AbstractGridType{T, false}; varname::String="ψ") where T<:Number
    if size(data,1)!=3
        @error("First dimension is not three")
    end
    data_sh = zeros(T,3,p.N_lats,p.N_lons)
    for i ∈ 1:3
        data_sh[i,:,:] = transform_grid(data[i,:,:], p, g; varname=varname)
    end
    return data_sh
end

function transform_grid(data::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::AbstractGridType{T, true}; varname::String="ψ") where T<:Number
    if size(data,1)!=3
        @error("First dimension is not three")
    end
    data_sh = CUDA.zeros(T,3,p.N_lats,p.N_lons)
    for i ∈ 1:3
        data_sh[i,:,:] = transform_grid(data[i,:,:], p, g; varname=varname)
    end
    return data_sh
end

transform_grid(data::AbstractArray{T,3}, m::QG3Model{T}; kwargs...) where T<:Number = transform_grid(data, m.p, m.g; kwargs...)
transform_grid(data::AbstractArray{T,4}, m::QG3Model{T}; kwargs...) where T<:Number = transform_grid(data, m.p, m.g; kwargs...)

function transform_grid(data::AbstractArray{T,4}, p::QG3ModelParameters{T}, g::AbstractGridType{T, true}; kwargs...) where T<:Number
    data_sh = CUDA.zeros(T,3, p.N_lats, p.N_lon, size(data,4))
    for it ∈ 1:size(data,4)
        data_sh[:,:,:,it] = transform_grid(data[:,:,:,it], p, g; varname=varname)
    end
    return data_sh
end

function transform_grid(data::AbstractArray{T,4}, p::QG3ModelParameters{T}, g::AbstractGridType{T, false}; kwargs...) where T<:Number
    data_sh = zeros(T,3, p.N_lats, p.N_lon,size(data,4))
    for it ∈ 1:size(data,4)
        data_sh[:,:,:,it] = transform_grid(data[:,:,:,it], p, g; varname=varname)
    end
    return data_sh
end
