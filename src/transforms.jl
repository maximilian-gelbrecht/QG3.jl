import Base.show

"""
    abstract type AbstractSHTransform{onGPU} 

Required fields for all subtypes: 

* outputsize

"""
abstract type AbstractSHTransform{onGPU} end

abstract type AbstractSHtoGridTransform{onGPU} <: AbstractSHTransform{onGPU} end 

abstract type AbstractGridtoSHTransform{onGPU} <: AbstractSHTransform{onGPU} end 


"""
    transform_SH(data::AbstractArray{T,N}, t::GaussianGridtoSHTransform) 
    transform_SH(data::AbstractArray, m::QG3Model) 

Transforms `data` into the spherical harmonics domain. The coefficents are ordered in a matrix in coloumns of the m value. On CPU the convention of FastTransform.jl is used (0, -1, 1, -2, 2, ...), on GPU the convention (0, 1, 2, 3, ...., (nothing, -1, -2, -3, ...)). Watch out, in future proabaly this might be standardized. 
"""
transform_SH

# CPU 4D Version (not meant for the model, but for data processing)
function transform_SH(data::AbstractArray{T,4}, t::AbstractGridtoSHTransform{false}) where T<:Number
    data_sh = zeros(T, size(data, 1), t.output_size..., size(data,4))
    for it ∈ 1:size(data,4)
        data_sh[:,:,:,it] = transform_SH(data[:,:,:,it], t)
    end
    return data_sh
end

# GPU 4D Version (not meant for the model, but for data processing)
function transform_SH(data::AbstractArray{T,4}, t::AbstractGridtoSHTransform{true}) where T<:Number
    data_sh = CUDA.zeros(T, size(data, 1), t.output_size..., size(data,4))
    for it ∈ 1:size(data,4)
        data_sh[:,:,:,it] = transform_SH(data[:,:,:,it], t)
    end
    return data_sh
end

transform_SH(data::AbstractArray, m::AbstractQG3Model) = transform_SH(data, m.g)
transform_SH(data::AbstractArray, g::AbstractGridType) = transform_SH(data, g.GtoSH)


"""
    transform_grid(data::AbstractArray{T,N}, t::SHtoGaussianGridTransform) 

Transforms `data` from the spherical harmonics domain to a Gaussian Grid. The coefficents are ordered in a matrix in coloumns of the m value. On CPU the convention of FastTransform.jl is used (0, -1, 1, -2, 2, ...), on GPU the convention (0, 1, 2, 3, ...., (nothing, -1, -2, -3, ...)). Watch out, in future proabaly this might be standardized. 

    transform_grid(data, m::QG3Model; varname="ψ")

If varname ∈ ["pv","q","vorticity"] the [1,1] element is set to zero first, as this element is set to the streamfunction-[1,1] usually in the ψtoq routines.
"""
transform_grid 

transform_grid(data::AbstractArray{T,N}, m::AbstractQG3Model; kwargs...) where {T,N} = transform_grid(data, m.g; kwargs...)
function transform_grid(data::AbstractArray{T,N}, g::AbstractGridType; varname::String="ψ") where {T,N}
    if varname ∈ ["pv","q","vorticity"]
        return transform_grid(setpvzero(data), g.SHtoG)
    else
        return transform_grid(data, g.SHtoG)
    end
end

function setpvzero(data::AbstractArray{T,2}) where {T}
    data_out = copy(data)
    data_out[1,1] = T(0)
    data_out
end

function setpvzero(data::AbstractArray{T,3}) where {T}
    data_out = copy(data)
    data_out[:,1,1] .= T(0)
    data_out
end 

function setpvzero(data::AbstractArray{T,4}) where {T}
    data_out = copy(data)
    data_out[:,1,1,:] .= T(0)
    data_out
end 

function transform_grid(data::AbstractArray{T,4}, t::AbstractSHtoGridTransform{true}) where {T}
   data_sh = CUDA.zeros(T, size(data,1), t.output_size..., size(data,4))
   for it ∈ 1:size(data,4)
       data_sh[:,:,:,it] = transform_grid(data[:,:,:,it], t)
   end
   return data_sh
end

function transform_grid(data::AbstractArray{T,4}, t::AbstractSHtoGridTransform{false}) where {T}
   data_sh =zeros(T, size(data,1), t.output_size..., size(data,4))
   for it ∈ 1:size(data,4)
       data_sh[:,:,:,it] = transform_grid(data[:,:,:,it], t)
   end
   return data_sh
end

"""
GaussianGridtoSHTransform(p::QG3ModelParameters{T}, N_level::Int=3)

Returns transform struct, that can be used with `transform_SH`. This is one is for a Gaussian Grid.
"""
struct GaussianGridtoSHTransform{P,S,T,U,V<:Union{AbstractVector,Nothing},TU,onGPU} <: AbstractGridtoSHTransform{onGPU}
    FT_2d::S
    FT_3d::T
    Pw::U
    truncate_array::V
    output_size::TU
end

show(io::IO, t::GaussianGridtoSHTransform{P,S,T,U,V,TU,true}) where {P,S,T,U,V,TU} = print(io, "Pre-computed Gaussian Grid to SH Transform{",P,"} on GPU")
show(io::IO, t::GaussianGridtoSHTransform{P,S,T,U,V,TU,false}) where {P,S,T,U,V,TU} = print(io, "Pre-computed Gaussian Grid to SH Transform{",P,"} on CPU")

function GaussianGridtoSHTransform(p::QG3ModelParameters{T}, N_level::Int=3) where {T}
    __, P = compute_P(p)
    Pw = compute_LegendreGauss(p, P)
    A_real = togpu(rand(T, N_level, p.N_lats, p.N_lons))

    if cuda_used[]
        Pw = reorder_SH_gpu(Pw, p)

        FT_2d = plan_cur2r(A_real[1,:,:], 2)
        FT_3d = plan_cur2r(A_real, 3)

        truncate_array = nothing
        outputsize = (p.N_lats, p.N_lons+2)
    else 
        FT_2d = FFTW.plan_r2r(A_real[1,:,:], FFTW.R2HC, 2)
        FT_3d = FFTW.plan_r2r(A_real, FFTW.R2HC, 3)

        m_p = 1:p.L
        m_n = p.N_lons:-1:p.N_lons-(p.L-2)

        truncate_array = [1]
        for im=1:(p.L-1)
            push!(truncate_array, m_n[im])
            push!(truncate_array, m_p[im+1])
        end
        outputsize = (p.L, p.M)
    end
    GaussianGridtoSHTransform{T,typeof(FT_2d),typeof(FT_3d),typeof(Pw), typeof(truncate_array), typeof(outputsize), cuda_used[]}(FT_2d, FT_3d, Pw, truncate_array, outputsize)
end

# 2D CPU version
function transform_SH(A::AbstractArray{P,2}, t::GaussianGridtoSHTransform{P,S,T,U,V,TU,false}) where {P,S,T,U,V,TU}
    FTA = (t.FT_2d * A)[:,t.truncate_array]
    @tullio out[il,im] := t.Pw[i,il,im] * FTA[i,im]
end

# 3D CPU version 
function  transform_SH(A::AbstractArray{P,3}, t::GaussianGridtoSHTransform{P,S,T,U,V,TU,false}) where {P,S,T,U,V,TU}
    FTA = (t.FT_3d * A)[:,:,t.truncate_array]
    @tullio out[ilvl,il,im] := t.Pw[ilat,il,im] * FTA[ilvl,ilat,im]
end

# 2D GPU version 
function transform_SH(A::AbstractArray{P,2}, t::GaussianGridtoSHTransform{P,S,T,U,V,TU,true}) where {P,S,T,U,V,TU}
    FTA = t.FT_2d * A

    # truncation is performed in this step as Pw has 0s where the expansion is truncated
    @tullio out[il,im] := t.Pw[i,il,im] * FTA[i,im]
end

# 3D GPU version
function transform_SH(A::AbstractArray{P,3}, t::GaussianGridtoSHTransform{P,S,T,U,V,TU,true}) where {P,S,T,U,V,TU}
    FTA = t.FT_3d * A

    # truncation is performed in this step as Pw has 0s where the expansion is truncated
    @tullio out[ilvl,il,im] := t.Pw[ilat,il,im] * FTA[ilvl,ilat,im]
end

struct SHtoGaussianGridTransform{R,S,T,U,TU,I<:Integer,onGPU} <: AbstractSHtoGridTransform{onGPU}
    iFT_2d::S
    iFT_3d::T
    P::U
    output_size::TU
    N_lats::I
    N_lons::I
    M::I
end

show(io::IO, t::SHtoGaussianGridTransform{P,S,T,U,TU,I,true}) where {P,S,T,U,TU,I} = print(io, "Pre-computed SH to Gaussian Grid Transform{",P,"} on GPU")
show(io::IO, t::SHtoGaussianGridTransform{P,S,T,U,TU,I,false}) where {P,S,T,U,TU,I} = print(io, "Pre-computed SH to Gaussian Grid Transform{",P,"} on CPU")

function SHtoGaussianGridTransform(p::QG3ModelParameters{T}, N_level::Int=3) where {T}
    __, P = compute_P(p)
    A_real = togpu(rand(T, N_level, p.N_lats, p.N_lons))

    if cuda_used[]
        P = reorder_SH_gpu(P, p)

        FT_2d = plan_cur2r(A_real[1,:,:], 2)
        iFT_2d = plan_cuir2r(FT_2d*(A_real[1,:,:]), p.N_lons, 2)

        FT_3d = plan_cur2r(A_real, 3)
        iFT_3d = plan_cuir2r(FT_3d*A_real, p.N_lons, 3)

    else 
        iFT_2d = FFTW.plan_r2r(A_real[1,:,:], FFTW.HC2R, 2)
        iFT_3d = FFTW.plan_r2r(A_real, FFTW.HC2R, 3)

    end
    outputsize = (p.N_lats, p.N_lons)

    SHtoGaussianGridTransform{T,typeof(iFT_2d),typeof(iFT_3d),typeof(P), typeof(outputsize), typeof(p.N_lats),cuda_used[]}(iFT_2d, iFT_3d, P, outputsize, p.N_lats, p.N_lons, p.M)
end

# 2D CPU Version 
function transform_grid(A::AbstractArray{P,2}, t::SHtoGaussianGridTransform{P,S,T,U,TU,I,false}) where {P,S,T,U,TU,I}

    out = batched_vec(t.P, A)

    # pad with zeros and adjust to indexing of FFTW
    t.iFT_2d * cat(out[:,1:2:end], zeros(P, t.N_lats, t.N_lons - t.M), out[:,end-1:-2:2], dims=2) ./ t.N_lons # has to be normalized as this is not done by FFTW
end

# 3D CPU Version
function transform_grid(A::AbstractArray{P,3}, t::SHtoGaussianGridTransform{P,S,T,U,TU,I,false}) where {P,S,T,U,TU,I}

    @tullio out[lvl, ilat, im] := t.P[ilat, il, im] * A[lvl, il, im]

    # pad with zeros and adjust to indexing of FFTW
    t.iFT_3d * cat(out[:,:,1:2:end], zeros(P, size(A,1), t.N_lats, t.N_lons - t.M), out[:,:,end-1:-2:2], dims=3) ./ t.N_lons # has to be normalized as this is not done by FFTW

end

# 2D GPU Version
function transform_grid(A::AbstractArray{P,2}, t::SHtoGaussianGridTransform{P,S,T,U,TU,I,true}) where {P,S,T,U,TU,I}

    out = batched_vec(t.P,A)
    
    t.iFT_2d * out
end

# 3D GPU Version
function transform_grid(A::AbstractArray{P,3}, t::SHtoGaussianGridTransform{P,S,T,U,TU,I,true}) where {P,S,T,U,TU,I}
    @tullio out[lvl, ilat, im] := t.P[ilat, il, im] * A[lvl, il, im]

    t.iFT_3d * out
end



"""

Pre-compute ass. Legendre Polynomials and dP/dx (derivative of ass. Legendre Polynomial) at the grid points and also the remainder of the Spherical Harmonics at the grid points using GSL

m values are stored 0,-1,1,-2,2,-3,3,... (on CPU)
m values are stored 0,1,2,3,4,5,6,7, ...l_max, 0 (nothing),-1, -2, -3, (on GPU)  (the second 0 is the Imanigary part / sin part of the fourier transform which is always identical to zero, it is kept here to have equal matrix sizes)

# so far only |m| is used, as I assume real SPH.


"""
function compute_P(L::Integer, M::Integer, μ::AbstractArray{T,1}; sh_norm=GSL_SF_LEGENDRE_SPHARM, CSPhase::Integer=-1,prefactor=false) where T<:Number

    N_lats = length(μ)
    P = zeros(T, N_lats, L, M)
    dPμdμ = zeros(T, N_lats, L, M)

    gsl_legendre_index(l,m) = m > l ? error("m > l, not defined") : sf_legendre_array_index(l,m)+1 # +1 because of c indexing vs julia indexing

    # normalization pre-factor for real SPH    
    pre_factor(m) = prefactor ? (m==0 ? T(1) : sqrt(T(2))) : T(1)

    for ilat ∈ 1:N_lats
        temp = sf_legendre_deriv_array_e(sh_norm, L - 1, μ[ilat], CSPhase)

        for m ∈ -(L-1):(L-1)
            for il ∈ 1:(L - abs(m)) # l = abs(m):l_max
                l = il + abs(m) - 1
                if m<0 # the ass. LP are actually the same for m<0 for our application as only |m| is needed, but I do this here in this way to have everything related to SH in the same matrix format
                    P[ilat, il, 2*abs(m)] = pre_factor(m) * temp[1][gsl_legendre_index(l,abs(m))]
                    dPμdμ[ilat, il, 2*abs(m)] = pre_factor(m) * temp[2][gsl_legendre_index(l,abs(m))]
                else
                    P[ilat, il, 2*m+1] = pre_factor(m) * temp[1][gsl_legendre_index(l,m)]
                    dPμdμ[ilat, il, 2*m+1] = pre_factor(m) * temp[2][gsl_legendre_index(l,m)]
                end
            end
        end
    end
    return dPμdμ, P
end
compute_P(p::QG3ModelParameters; kwargs...) = compute_P(p.L, p.M, p.μ; kwargs...)

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
        P[i,:,:] *= (2π*w[i]) # 4π from integral norm 
    end
    P
end

function compute_LegendreGauss(p::QG3ModelParameters{T}, P::AbstractArray{T,3}; reltol::Number=1e-2) where T<:Number
    w = compute_GaussWeights(p, reltol)
    return compute_LegendreGauss(p, P, w)
end