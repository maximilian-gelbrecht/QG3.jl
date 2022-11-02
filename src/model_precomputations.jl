# pre-computations are only meant to be performed on CPU, not GPU

"""
Pre-compute drag coefficient k in real grid space, double check with values from paper
"""
function compute_k(p::QG3ModelParameters{T}) where T<:Number
    k = zeros(T, p.N_lats, p.N_lons)
    LS = tocpu(p.LS)
    h = tocpu(p.h)
    for i ∈ 1:p.N_lats
        for j ∈ 1:p.N_lons
            k[i,j] = p.τEi * (T(1) + p.α1*LS[i,j] + p.α2*FH.(h[i,j]))
        end
    end
    return k
end
FH(h::T) where T<:Number =  T(1) - exp(-h/T(1000))


"""
Pre-compute matrices involved in transforming vorticity to streamfunction and back

q = Tψq * ψ + f
ψ = Tqψ * (q - f) = Tψq^(-1) * (q - f)

# For l=0 the matrix is singular. In the fortran code they set q to something which for me does not make sense. Here Tψq(l=0) is just 1, so that transforming back and forth recovers the correct ψ. This makes sense in my eyes because q is only ever used for its spatial derivatives, all of which are 0 for l=0.

"""
function compute_ψq_transform_matrices(p::QG3ModelParameters{T}, Δ) where T<:Number
    Tqψ = zeros(T, p.L, 3,3)
    Tψq = zeros(T, p.L, 3,3)

    # when denoting the transform for l=0, m=0 we get a singular matrix that's not invertible. right now we just set to always hand over the identity, the only term in the whole model that modifies (0,0) element is the temperature relaxation as all other work on derivaties (which are zero for the constant field)
    Tψq[1,1,1] = T(1)
    Tψq[1,2,2] = T(1)
    Tψq[1,3,3] = T(1)
    Tqψ[1,:,:] = inv(Tψq[1,:,:])
    for l ∈ 2:p.L # this is actually l=(l+1) in terms of SH numbers due to 1-indexing
        Tψq[l,1,1] = Δ[l,1] - p.R1i
        Tψq[l,1,2] = p.R1i

        Tψq[l,2,1] = p.R1i
        Tψq[l,2,2] = Δ[l,1] - p.R1i - p.R2i
        Tψq[l,2,3] = p.R2i

        Tψq[l,3,2] = p.R2i
        Tψq[l,3,3] = Δ[l,1] - p.R2i

        Tqψ[l,:,:] = inv(Tψq[l,:,:])
    end

    return Tqψ, Tψq
end

"""
    compute_batched_ψq_transform_matrices

prepares the transform from q to ψ and back for a batched matrix vector multiply, see also (@ref)[`compute_ψq_transform_matrices`]
"""
function compute_batched_ψq_transform_matrices(p::QG3ModelParameters{T}, Δ) where T<:Number
    Tqψ, Tψq = compute_ψq_transform_matrices(p, Δ)
    return compute_batched_ψq_transform_matrices(p, Tqψ, Tψq)
end

function compute_batched_ψq_transform_matrices(p::QG3ModelParameters{T}, Tqψ, Tψq) where T<:Number

    if cuda_used[]
        bTψq = zeros(T,3,3,p.N_lats,p.N_lons+2)
        bTqψ = zeros(T,3,3,p.N_lats,p.N_lons+2)

        for m ∈ -(p.L-1):(p.L-1)
            for il ∈ 1:(p.L - abs(m))
                l = il + abs(m) - 1
                im = m < 0 ? abs(m) + Int(p.N_lons/2) + 2 : m + 1
                bTψq[:,:,il,im] = Tψq[l+1,:,:]
                bTqψ[:,:,il,im] = Tqψ[l+1,:,:]
            end
        end

        return togpu(reshape(bTqψ,3,3,:)), togpu(reshape(bTψq,3,3,:))
    else
        bTψq = zeros(T,3,3,p.L,p.M)
        bTqψ = zeros(T,3,3,p.L,p.M)

        for m ∈ -(p.L-1):(p.L-1)
            for il ∈ 1:(p.L - abs(m))
                l = il + abs(m) - 1
                im = m<0 ? 2*abs(m) : 2*m+1
                bTψq[:,:,il,im] = Tψq[l+1,:,:]
                bTqψ[:,:,il,im] = Tqψ[l+1,:,:]
            end
        end
        return reshape(bTqψ,3,3,:), reshape(bTψq,3,3,:)
    end
end


"""
Pre-compute matrices involved in the temperature relaxation in matrix form (for GPU / 3d fields)

TR = TR_Matrix * ψ  ∀ l ∈ [0,l_max]

# For l=0 the matrix is set to be zero

"""
function compute_TR_matrix(p::QG3ModelParameters{T}) where T<:Number

    TR_matrix = zeros(T, p.L, 3,3)

    for l ∈ 2:p.L # this is actually l=(l+1) in terms of SH numbers due to 1-indexing
        TR_matrix[l,1,1] = - p.R1i * p.τRi
        TR_matrix[l,1,2] = p.R1i * p.τRi

        TR_matrix[l,2,1] = p.R1i * p.τRi
        TR_matrix[l,2,2] = (- p.R1i - p.R2i) * p.τRi
        TR_matrix[l,2,3] = p.R2i * p.τRi

        TR_matrix[l,3,2] = p.R2i * p.τRi
        TR_matrix[l,3,3] = - p.R2i * p.τRi
    end

    return TR_matrix
end

"""
    compute_batched_TR_matrix

prepares the TR = TR_Matrix * ψ  ∀ l ∈ [0,l_max] matrix for batched multiply
"""
function compute_batched_TR_matrix(p::QG3ModelParameters{T}) where T<:Number
    TR_matrix = compute_TR_matrix(p)
    return compute_batched_TR_matrix(p, TR_matrix)
end

function compute_batched_TR_matrix(p::QG3ModelParameters{T}, TR::AbstractArray{T,3}) where T<:Number

    if cuda_used[]
        bTR = zeros(T,3,3,p.N_lats,p.N_lons+2)

        for m ∈ -(p.L-1):(p.L-1)
            for il ∈ 1:(p.L - abs(m))
                l = il + abs(m) - 1
                im = m < 0 ? abs(m) + Int(p.N_lons/2) + 2 : m + 1
                bTR[:,:,il,im] = TR[l+1,:,:]
            end
        end

        return togpu(reshape(bTR,3,3,:))
    else
        bTR = zeros(T,3,3,p.L,p.M)

        for m ∈ -(p.L-1):(p.L-1)
            for il ∈ 1:(p.L - abs(m))
                l = il + abs(m) - 1
                im = m<0 ? 2*abs(m) : 2*m+1
                bTR[:,:,il,im] = TR[l+1,:,:]
            end
        end
        return reshape(bTR,3,3,:)
    end
end


"""
Pre-compute a matrix with with the Coriolis factor
"""
function compute_coriolis_vector_grid(p::QG3ModelParameters{T}) where T<:Number
    f = zeros(T, 3 , p.N_lats, p.N_lons)
    lats = tocpu(p.lats)
    h = tocpu(p.h)

    for ilat ∈ 1:p.N_lats
        f[1:2,ilat,:] .= sin(lats[ilat])

        for ilon ∈ 1:p.N_lons
            f[3,ilat,ilon] = sin(lats[ilat])*(T(1) + h[ilat,ilon]/p.H0)
        end
    end
    return f
end

"""
Pre-compute the additional contribution of the Coriolis force to the 850hPa Jacobian component in GPU ready format, for the Jacobian at 850hPa, q = q' + f(1+h/H_0) = q' + f + f*h/H_0, so that the thrid term has to be added.
"""
function compute_f_J3(p::QG3ModelParameters{T}, f::AbstractArray{T,3}) where T<:Number

    f_J3 = zeros(T, 3, size(f,2), size(f,3))
    f_J3[3,:,:] = f[3,:,:] - f[2,:,:]

    return f_J3
end

"""
Pre-compute the Laplacian in Spherical Harmonics, follows the matrix convention of FastTransforms.jl
"""
function compute_Δ(p::QG3ModelParameters{T}) where T<:Number
    l = T.(lMatrix(p))
    return -l .* (l .+ 1)
end

"""
Pre-compute the inverse Laplacian in Spherical Harmonics, follows the matrix convention of FastTransforms.jl
"""
function compute_Δ⁻¹(p::QG3ModelParameters{T}) where T<:Number
    Δ⁻¹ = inv.(compute_Δ(p))
    Δ⁻¹[isinf.(Δ⁻¹)] .= T(0) # set integration constant and spurious elements zero 
    return Δ⁻¹
end


""""
Pre-compute the 8-th derivative in Spherical Harmonics
"""
function compute_∇8(p::QG3ModelParameters)
    Δ = compute_Δ(p)
    return Δ.*Δ.*Δ.*Δ
end

"""
Pre-compute array of temperature relaxation coefficients.

(l=0, m=0)-coefficient is assigned zero. This results in this expansion coefficient beeing constant in the whole model.
"""
function compute_TR(p::QG3ModelParameters{T}) where T<:Number
    TRcoeffs = zeros(T, 2, p.L, p.M)

    TRcoeffs[1,:,:] .= p.τRi * p.R1i
    TRcoeffs[2,:,:] .= p.τRi * p.R2i

    TRcoeffs[1,1,1] = T(0)
    TRcoeffs[2,1,1] = T(0)

    return TRcoeffs
end

"""
Pre-compute cos(ϕ) (latitude) matrix
"""
function compute_cosϕ(p::QG3ModelParameters{T}) where {T}
    return reshape(T.(cos.(tocpu(p.lats))), p.N_lats, 1)
end

"""
Pre-compute cos(ϕ)^-1 (latitude matrix)
"""
compute_acosϕi(p::QG3ModelParameters) = compute_cosϕ(p).^(-1)
