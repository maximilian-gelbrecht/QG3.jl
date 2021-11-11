
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

"""
Pre-compute a matrix with with the Coriolis factor
"""
compute_coriolis_vector_SH(p::QG3ModelParameters, SHPlan, FTPlan, P) = transform_SH(compute_coriolis_vector_grid(p), SHPlan, FTPlan, P)

function compute_coriolis_vector_grid(p::QG3ModelParameters{T}) where T<:Number
    f = zeros(T, 3 , p.N_lats, p.N_lons)
    lats = tocpu(p.lats)
    h = tocpu(p.h)

    for ilat ∈ 1:p.N_lats
        f[1:2,ilat,:] .= T(2)*p.Ω*sin(lats[ilat])

        for ilon ∈ 1:p.N_lons
            f[3,ilat,ilon] = T(2)*p.Ω*sin(lats[ilat])*(T(1) + h[ilat,ilon]/p.H0)
        end
    end
    return f
end

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

"""
Pre-compute the Laplacian in Spherical Harmonics, follows the matrix convention of FastTransforms.jl
"""
function compute_Δ(p::QG3ModelParameters{T}) where T<:Number
    l = lMatrix(p)
    return -l .* (l .+ 1) ./ (p.a^2)
end


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

    TRcoeffs[1,2:end,2:end] .= p.τRi * p.R1i
    TRcoeffs[2,2:end,2:end] .= p.τRi * p.R2i

    return TRcoeffs
end



"""
Pre-compute cos(ϕ) (latitude) matrix
"""
function compute_cosϕ(p::QG3ModelParameters)
cosϕ = zeros(Float64, p.N_lats, p.N_lons)
lats = tocpu(p.lats)
for i=1:p.N_lats
    cosϕ[i,:] .= cos(lats[i])
end

return cosϕ
end

"""
Pre-compute (a*cos(ϕ))^-1 (latitude matrix)
"""
compute_acosϕi(p::QG3ModelParameters) = (compute_cosϕ(p) .* p.a).^(-1)
