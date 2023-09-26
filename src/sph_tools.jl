# this file contains function that compute Spherical Harmonics
# Right now there are three different variants possible
# a) A self implemented Gaussian Grid based version
# b) Fast Transforms.jl for regular grids
#

# shared functions

"""
Reorders the SH coefficient so that computations on GPU are more efficient, it also includes the truncation as the SH array has size N_lat x N_lon, however all entries outside of L x M are zeroes.

Original order (FastTransforms.jl) columns by m : 0, -1, 1, -2, -2, ....

New order columns by m: 0, 1, 2, ... l_max, 0 (nothing), -1, -2, ..

2d input assumes L x M matrix, (all fields in SH), also enlarges the matrix to include truncation (additional elements are zero) to N_lat x N_lons
3d input assumes N_lat (or 3) x L x M matrix (e.g. precomputed legendre polynomials), also enlarges the matrix to include truncation (additional elements are zero) to N_lat (or 3) x N_lats x N_lons

Incase CUDA is not used, it just return the input.
"""
function reorder_SH_gpu(A::AbstractArray{S,2}, p::QG3ModelParameters{T}) where {S,T}
    if !(cuda_used[])
        return A
    end

    reindex = [1:2:(p.N_lons+2);[(p.N_lons+2)]; 2:2:(p.N_lons+1)] # the middle one is the 0 (nothing)
    out = zeros(S, p.N_lats, p.N_lons+2)
    out[1:p.L, 1:p.M] = A
    return togpu(out[:,reindex])
end

function reorder_SH_gpu(A::AbstractArray{S,3}, p::QG3ModelParameters{T}) where {S,T}
    if !(cuda_used[])
        return A
    end

    reindex = [1:2:(p.N_lons+2);[(p.N_lons+2)]; 2:2:(p.N_lons+1)]

    out = zeros(S, size(A, 1), p.N_lats, p.N_lons+2)
    out[:, 1:p.L, 1:p.M] = A
    return togpu(out[:,:,reindex])
end

function reorder_SH_gpu(A::AbstractArray{S,4}, p::QG3ModelParameters{T}) where {S,T}
    if !(cuda_used[])
        return A
    end

    reindex = [1:2:(p.N_lons+2);[(p.N_lons+2)]; 2:2:(p.N_lons+1)]

    out = zeros(S, size(A, 1), p.N_lats, p.N_lons+2, size(A,4))
    out[:, 1:p.L, 1:p.M, :] = A
    return togpu(out[:,:,reindex,:])
end


"""
    reorder_SH_cpu(A::AbstractArray{T,2},p::QG3ModelParameters)

Reorders the SH coefficient so that computations on CPU are more efficient, inverse of [`reorder_SH_gpu`](@ref)
"""
function reorder_SH_cpu(A::AbstractArray{T,2},p::QG3ModelParameters) where T
    @assert size(A) == (p.N_lats, p.N_lons+2) "Wrong array size, probably not GPU SH array"
    
    Nlons2 = Int((p.N_lons+2)/2)
    reindex = collect(Iterators.flatten(zip(1:Nlons2,Nlons2+2:p.N_lons+2)))
    
    out = Array(A)[1:p.L, reindex]
    return out[:,1:p.M]
end 

"""
    reorder_SH_cpu(A::AbstractArray{T,3},p::QG3ModelParameters)

Reorders the SH coefficient so that computations on CPU are more efficient, inverse of [`reorder_SH_gpu`](@ref)
"""
function reorder_SH_cpu(A::AbstractArray{T,3},p::QG3ModelParameters) where T
    @assert size(A,2) == p.N_lats "Wrong array size, probably not GPU SH array"
    @assert size(A,3) == p.N_lons+2 "Wrong array size, probably not GPU SH array"
    
    Nlons2 = Int((p.N_lons+2)/2)
    reindex = collect(Iterators.flatten(zip(1:Nlons2,Nlons2+2:p.N_lons+2)))
    
    out = Array(A)[:,1:p.L, reindex]
    return out[:,:,1:p.M]
end 

"""
    reorder_SH_cpu(A::AbstractArray{T,4},p::QG3ModelParameters)

Reorders the SH coefficient so that computations on CPU are more efficient, inverse of [`reorder_SH_gpu`](@ref)
"""
function reorder_SH_cpu(A::AbstractArray{T,4},p::QG3ModelParameters) where T
    @assert size(A,2) == p.N_lats "Wrong array size, probably not GPU SH array"
    @assert size(A,3) == p.N_lons+2 "Wrong array size, probably not GPU SH array"
    
    Nlons2 = Int((p.N_lons+2)/2)
    reindex = collect(Iterators.flatten(zip(1:Nlons2,Nlons2+2:p.N_lons+2)))
    
    out = Array(A)[:,1:p.L, reindex,:]
    return out[:,:,1:p.M,:]
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
    change_msign!(A)

Change the sign of the m in SH (FastTranforms.jl convention of storing them). This version swaps the columns inplace
"""
function change_msign!(A)
    for i=1:(size(A,1)-1)
        A[:,2*i], A[:,2*i+1] = A[:,2*i+1], A[:,2*i]
    end
    A
end

"""
    lMatrix(L, M; GPU=nothing)

Return l-Matrix of SH coefficients in convention of FastTransforms.jl. 
"""
function lMatrix(L::Integer, M::Integer)
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
    mMatrix(L::Integer, M::Integer)

Pre-compute a matrix with (m) values of the SH matrix format of FastTransforms.jl, used for zonal derivative

Kwarg 'GPU', if given, overrides the automatic detection of wheather or not a GPU is avaible. 
"""
function mMatrix(L::T, M::T) where T<:Integer
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
mMatrix(p::QG3ModelParameters{T}) where T<:Number = T.(mMatrix(p.L,p.M))
mMatrix(p::QG3Model) = mMatrix(p.p)

# test this
# Y_lm(π - θ, ϕ) = (-1)^(l+m) Y_lm(θ, ϕ)
# thus even l+m symmetric, odd l+m anti-symmetric
function symmetrize_equator!(ψ::AbstractArray{T,2}, p::QG3ModelParameters{T}) where T<:Number

    m = mMatrix(p)
    l = lMatrix(p)

    ψ[mod.(l + abs.(m), 2) .== 0] .= T(0)
end

function symmetrize_equator!(ψ::AbstractArray{T,3}, p::QG3ModelParameters{T}) where T<:Number
    for i=1:3
        symmetrize_equator!(ψ[i,:,:], p)
    end
end

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

"""
    SH_zero_mask(p::QG3.QG3ModelParameters, size_tup=nothing)

Returns a mask that zeroes out all spurious elements of the SH coefficients + the (0,0) element which is also set zero
"""
function SH_zero_mask(p::QG3.QG3ModelParameters, size_tup=nothing; N_batch::Int=0)
    mask = QG3.lMatrix(p) .!= 0
    mask[1,1] = 1

    if isnothing(size_tup)
        return mask 
    end 

    if (N_batch > 0) & !(isnothing(size_tup))
        @assert size_tup[4]==N_batch 
        
        mask_out = zeros(Bool, size_tup[1], size(mask,1), size(mask,2), size_tup[4])

        for i=1:size_tup[1]
            for j=1:size_tup[4]
                mask_out[i,:,:,j] = mask 
            end 
        end 
        
        return mask_out
    end 
    
    if length(size_tup)==2 # regular SPH matrix 
        return mask 
    elseif length(size_tup)==3 # lvl x SPH x SPH 
        return cat([reshape(mask, 1, size(mask)...) for i=1:size_tup[1]]..., dims=1)
    elseif length(size_tup)==4 # size_tup[1] x size_tup[2] x SPH X SPH 
        mask = cat([reshape(mask, 1, size(mask)...) for i=1:size_tup[2]]..., dims=1)
        return mask = cat([reshape(mask, 1, size(mask)...) for i=1:size_tup[1]]..., dims=1)
    else
        error("Wrong dimension of size should be 2,3 or 4")
    end
end