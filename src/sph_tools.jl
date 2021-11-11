# this file contains function that compute Spherical Harmonics
# Right now there are three different variants possible
# a) A self implemented Gaussian Grid based version
# b) Fast Transforms.jl for regular grids
# c) Spherepack via pyspharm. c) should only be used as a comparision and to check that everything is working as expected
#
#




# shared functions

"""

Pre-compute ass. Legendre Polynomials and dP/dx (derivative of ass. Legendre Polynomial) at the grid points and also the remainder of the Spherical Harmonics at the grid points using GSL

m values are stored 0,-1,1,-2,2,-3,3,...

# double check -m values , so far only |m| is used, as I assume real SPH.


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
"""
change_msign(A::AbstractArray{T,2}, swap_array) where T<:Number = view(A,:,swap_array)

function change_msign(A::AbstractArray{T,3}, i::Int) where T<:Number
    arr = [1;vcat([[2*i+1,2*i] for i=1:size(A,2)-1]...)]
    _change_msign(A, i, arr)
end

_change_msign(A::AbstractArray{T,3}, i, arr) where T<:Number = @inbounds view(A,i,:,arr)



shift_L_minus(ψ::AbstractArray{T,2}) where T<:Number = vcat(transpose(zeros(T,size(ψ,2))), ψ[1:end-1,:])

shift_L_minus(ψ::AbstractArray{T,3}) where T<:Number = cat(zeros(T,1,size(ψ,2),size(ψ,3)), ψ[1:end-1,:,:],dims=1)

shift_L_plus(ψ::AbstractArray{T,2}) where T<:Number = vcat(ψ[2:end,:], transpose(zeros(T,size(ψ,2))))

shift_L_plus(ψ::AbstractArray{T,3}) where T<:Number = cat(ψ[2:end,:,:],zeros(T,1,size(ψ,2),size(ψ,3)),dims=1)

function set_Lval_zero(ψ::AbstractArray{T,2}, l::Int, p::QG3ModelParameters) where T<:Number
    o = copy(ψ)
    lM = lMatrix(p)
    o[lM .== l] .= 0
    return o
end

function set_Lval_zero(ψ::AbstractArray{T,3}, l::Int, p::QG3ModelParameters) where T<:Number
    o = similar(ψ)
    for i=1:size(ψ,3)
        o[:,:,i] = set_Lval_zero(ψ[:,:,i],l,p)
    end
    return o
end

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

sphφevaluateDFT(φ::T, M::Integer, N_lons::Integer) where T<:Number = T(2/N_lons)*(M ≥ 0 ? cos(M*φ) : sin(-M*φ))


"""
manually implemented transform from spherical harmonics to grid. This works for all grid types, the grid type is indirectly specified through the pre-computed ass. legendre polynomials

CPU variant
"""
function transformSHtoGGrid(A::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T, false}) where T<:Number

    out = batched_vec(g.P,A)

    # pad with zeros and adjust to indexing of FFTW
    g.iFT * cat(out[:,1:2:end], zeros(T, p.N_lats, p.N_lons - p.M), out[:,end-1:-2:2], dims=2)
end

# GPU/CUDA variant
function transformSHtoGGrid(A::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T, true}) where T<:Number
    out = batched_vec(g.P,A)

    # pad with zeros and adjust to complex valued of CUDA FFT
    g.iFT * complex.(cat(out[:,1:2:end], CUDA.zeros(T, p.N_lats, div(p.N_lons,2) + 1 - p.L), dims=2), cat(CUDA.zeros(T,p.N_lats,1), out[:,2:2:end], CUDA.zeros(T, p.N_lats, div(p.N_lons,2) + 1 - p.L), dims=2))
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

GPU variant, r2c fft instead of r2r fft
"""
function transformGGridtoSH(A::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::GaussianGrid{T, true}) where T<:Number

    FTA = g.FT * A

    # deal with the complex array, turn it into half complex format and truncate
    FTA_HC = reinterpret(T, FTA)
    Re_FTA = @view FTA_HC[1:2:(end-1), 1:p.L]
    Im_FTA = @view FTA_HC[2:2:end, 2:p.L]

    # into the convention of fasttransform.jl
    HCr = cat(Re_FTA, Im_FTA, dims=2)[:,g.truncate_array]

    @tullio out[il,im] := g.Pw[i,il,im] * HCr[i,im]
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
