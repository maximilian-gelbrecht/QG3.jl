# this file contains all the code to take derivatives

# derivate functions follow the naming scheme: "Domain1Input"to"Domain2Output"_d"derivativeby"

# there are variants for GPU and CPU, the different grids and 3d and 2d fields


"""
    make3d(A::AbstractArray{T,2})

    repeats an array three times to turn it into (3 x size(A,1) x size(A,2)), for the fully matrix version of model.
"""
function make3d(A::AbstractArray{T,2}) where T<:Number
    out = zeros(T, 3, size(A,1), size(A,2))

    for i=1:3
        out[i,:,:] = A
    end

    togpu(out)
end


"""
derivative of input after φ (polar angle) or λ (longtitude) in SH to Grid, only for a single layer
"""
SHtoGrid_dφ(ψ, m::QG3Model{T}) where T<:Number = transform_grid(SHtoSH_dφ(ψ,m), m)

SHtoGrid_dφ(ψ::AbstractArray{T,3}, i::Integer, m::QG3Model{T}) where T<:Number = transform_grid(SHtoSH_dφ(ψ, i, m), m)
SHtoGrid_dλ(ψ, m) = SHtoGrid_dφ(ψ, m)
SHtoGrid_dλ(ψ, i, m) = SHtoGrid_dφ(ψ, i, m)


"""
derivative of input after φ (polar angle/longtitude) in SH, output in SH
"""
SHtoSH_dφ(ψ, m::QG3Model{T}) where T<:Number = SHtoSH_dφ(ψ, m.g)

SHtoSH_dφ(ψ::AbstractArray{T,2}, g::AbstractGridType{T}) where T<:Number = _SHtoSH_dφ(ψ, g.mm, g.swap_m_sign_array)

# 3d field variant
SHtoSH_dφ(ψ::AbstractArray{T,3}, g::AbstractGridType{T}) where T<:Number = _SHtoSH_dφ(ψ, g.mm_3d, g.swap_m_sign_array)

_SHtoSH_dφ(ψ, mm, swap_arr) where T<:Number = mm .* change_msign(ψ, swap_arr)


"""
derivative of input after φ (polar angle/longtitude) in SH, output in SH

these are  variants with AbstractArray{T,3} and index i to select which layer is the input for the derivative.

there is currently a bug or at least missing feature in Zygote, the AD library, that stops views from always working flawlessly when a view is mixed with prior indexing of an array. We need a view for the derivative after φ to change the sign of m, so here is a differentiable variant of the SHtoSH_dφ function
"""
SHtoSH_dφ(ψ::AbstractArray{T,3}, i::Integer, m::QG3Model{T}) where T<:Number = _SHtoSH_dφ(ψ, i, m.g.mm, m.g.swap_m_sign_array)

_SHtoSH_dφ(ψ::AbstractArray{T,3}, i::Integer, mm::AbstractArray{T,2}, swap_array::AbstractArray{Int,1}) where T<:Number = mm .* change_msign(ψ, i, swap_array)

SHtoSH_dλ(ψ, m) = SHtoSH_dφ(ψ, m)
SHtoSH_dλ(ψ, i, m) = SHtoSH_dφ(ψ, i, m)


"""
derivative of input after θ (azimutal angle/colatitude) in SH, uses pre computed SH evaluations (dependend on the grid type)
"""
SHtoGrid_dθ(ψ, m::QG3Model{T}) where T<:Number = SHtoGrid_dθ(ψ, m.p, m.g)

SHtoGrid_dθ(ψ::AbstractArray{T,2}, p::QG3ModelParameters{T}, g::Union{GaussianGrid{T},RegularGrid{T}}) where T<:Number = g.msinθ .* SHtoGrid_dμ(ψ, p, g)

SHtoGrid_dθ(ψ::AbstractArray{T,3}, p::QG3ModelParameters{T}, g::Union{GaussianGrid{T},RegularGrid{T}}) where T<:Number = g.msinθ_3d .* SHtoGrid_dμ(ψ, p, g)



"""
derivative of input after μ = sinϕ in SH, uses pre computed SH evaluations
"""
SHtoGrid_dμ(ψ, m::QG3Model{T}) where T<:Number = SHtoGrid_dμ(ψ, m.p, m.g)

SHtoGrid_dμ(ψ, p::QG3ModelParameters{T}, g::GaussianGrid{T}) where T<:Number = _SHtoGrid_dμθ(ψ, g.dPμdμ, p, g)

SHtoGrid_dμ(ψ, p::QG3ModelParameters{T}, g::RegularGrid{T}) where T<:Number = _SHtoGrid_dμθ(ψ, g.dPμdμ, p, g)

"""
Performs the latitude-based derivates. Uses a form of synthesis based on pre-computed values of the ass. Legendre polynomials at the grid points.

This version is optimized to be non-mutating with batched_vec. The inner batched_vec correspodends to an inverse Legendre transform and the outer multiply to an inverse Fourier transform.
"""
function _SHtoGrid_dμθ(ψ::AbstractArray{T,2}, dP::AbstractArray{T,3}, p::QG3ModelParameters, g::AbstractGridType{T, false}) where T<:Number

    out = batched_vec(dP, ψ)

    g.iFT * cat(out[:,1:2:end], zeros(T, p.N_lats, p.N_lons - p.M), out[:,end-1:-2:2], dims=2) ./ p.N_lons # has to be normalized as this is not done by FFTW
end

function _SHtoGrid_dμθ(ψ::AbstractArray{T,2}, dP::AbstractArray{T,3}, p::QG3ModelParameters, g::AbstractGridType{T, true}) where T<:Number

    @tullio out[ilat, im] := dP[ilat, il, im] * ψ[il, im]

    Re = @view out[:,1:(Int(p.N_lons/2)+1)]
    Im = @view out[:,(Int(p.N_lons/2)+2):end]

    g.iFT * complex.(Re, Im)
end

# 3d field CPU
function _SHtoGrid_dμθ(ψ::AbstractArray{T,3}, dP::AbstractArray{T,3}, p::QG3ModelParameters, g::AbstractGridType{T, false}) where T<:Number

    @tullio out[lvl, ilat, im] := dP[ilat, il, im] * ψ[lvl, il, im]

    g.iFT_3d * cat(out[:,:,1:2:end], zeros(T, 3, p.N_lats, p.N_lons - p.M), out[:,:,end-1:-2:2], dims=3) ./ p.N_lons # has to be normalized as this is not done by FFTW
end

# 3d field GPU
function _SHtoGrid_dμθ(ψ::AbstractArray{T,3}, dP::AbstractArray{T,3}, p::QG3ModelParameters, g::AbstractGridType{T, true}) where T<:Number

    @tullio out[ilvl, ilat, im] := dP[ilat, il, im] * ψ[ilvl, il, im]

    Re = @view out[:,:,1:(Int(p.N_lons/2)+1)]
    Im = @view out[:,:,(Int(p.N_lons/2)+2):end]

    g.iFT_3d * complex.(Re, Im)
end

SHtoSH_dθ(ψ,m) = transform_SH(SHtoGrid_dθ(ψ,m), m)
SHtoSH_dϕ(ψ,m) = eltype(ψ)(-1) .* SHtoSH_dθ(ψ, m)
SHtoGrid_dϕ(ψ,m) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, m)
SHtoGrid_dϕ(ψ,p,g) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, p, g)