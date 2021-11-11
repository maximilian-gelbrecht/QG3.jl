"""
Pre-compute Forcing S data level,X,Y,t

Compute the forcing according to Roads, this is outlined in the MM and Corti paper as well.

This uses an unfiltered version, in the Corti paper there is a filter. Maybe implement it as well

* data, input data
* m, QG3Model Parameters and pre-computed fields
* kwargs
    * datasource, either 'qprime' (anomalous potential vorticity) or 'sf' (streamfunction)
"""
function compute_S_Roads(data::AbstractArray{T,4}, m::QG3Model{T}; datasource="ψ") where T<:Number
    Nt = size(data,4)

    if datasource ∈ ["sf","ψ"]
        # get avg climatalogy
        ψc = dropdims(mean(data, dims=4),dims=4)
        Δq = similar(data)
        for it=1:Nt
            Δq[:,:,:,it] = ψtoqprime(m, data[:,:,:,it])
        end
        qc = dropdims(mean(Δq, dims=4),dims=4)
        Δψ = similar(data)
        #get deviation from average
        for it=1:Nt
            Δψ[:,:,:,it] = data[:,:,:,it] - ψc
            Δq[:,:,:,it] = Δq[:,:,:,it] - qc
        end

    elseif datasource=="qprime"

        qc = dropdims(mean(data, dims=4),dims=4)
        Δψ = similar(data)

        for it=1:Nt
            Δψ[:,:,:,it] = qprimetoψ(m, data[:,:,:,it])
        end
        ψc = dropdims(mean(Δψ, dims=4),dims=4)
        Δq = similar(data)
        #get deviation from average
        for it=1:Nt
            Δq[:,:,:,it] = data[:,:,:,it] - qc
            Δψ[:,:,:,it] = Δψ[:,:,:,it] - ψc
        end
    else
        error("Unknown datasource, should be 'qprime' or 'sf'.")
    end

    Jv = similar(data[:,:,:,1])  # doing it like this is agnostic to whether or not it is on GPU or CPU
    Jv .= 0 # avg Jacobian variability
    for it=1:Nt
        Jv[1,:,:] += J_F(Δψ[1,:,:,it], Δq[1,:,:,it], m)
        Jv[2,:,:] += J_F(Δψ[2,:,:,it], Δq[2,:,:,it], m)
        Jv[3,:,:] += J_F(Δψ[3,:,:,it], Δq[3,:,:,it], m)
    end
    Jv ./= Nt

    return permutedims(cat(
    J(ψc[1,:,:], qc[1,:,:], m) + Jv[1,:,:] + D1(ψc, qc, m),
    J(ψc[2,:,:], qc[2,:,:], m) + Jv[2,:,:] + D2(ψc, qc, m),
    J3(ψc[3,:,:], qc[3,:,:], m) + Jv[3,:,:] + D3(ψc, qc, m),
    dims=3),[3,1,2])
end

"""
    TimeForcing

Time-dependent Forcing
"""
struct ContinousTimeForcing
    S_winter::AbstractArray
    S_summer::AbstractArray
    ν # Frequency
    Nt # Number of Time steps
end

function (S::ContinousTimeForcing)(t::Number)
    return 0.5*(sin(S.ν*t + π/2)+1) .* S.S_winter + 0.5*(sin(S.ν*t - π/2)+1) .* S.S_summer
end


struct DiscreteTimeForcing
    S::AbstractArray
    Nt # number of time steps
end

function DiscreteTimeForcing(S::ContinousTimeForcing)
    F = zeros(eltype(S.S_winter),size(S.S_winter,1), size(S.S_winter, 2), size(S.S_winter, 3), S.Nt)

    for t ∈ 0:(S.Nt-1)
        F[:,:,:,t+1] = S(t)
    end
    return DiscreteTimeForcing(F, S.Nt)
end

(S::DiscreteTimeForcing)(t::Number) = S.S[:,:,:,Int(ceil(t%S.Nt))]







"""
Pre-compute time-dependent Forcing S data level,X,Y,t

Like in the Corti paper for the seasonal run this interpolates between two extremal forcings.

* Nt_year: how many days does a year have / period length of the sinus
"""
function compute_S_t_corti(data, summer_ind, winter_ind, m::QG3Model, Nt_year::Int=365, time_unit="s", datasource="sf")

    S_winter = compute_S(data[:,:,:,winter_ind], m, datasource=datasource)
    S_summer = compute_S(data[:,:,:,summer_ind], m, datasource=datasource)

    ν=2π/Nt_year


end
"""

data = ψ_SH[:,:,:,1:60]
datasource = "sf"
m = qg3p

Nt = size(data,4)

if datasource=="sf"
    # get avg climatalogy
    ψc = dropdims(mean(data, dims=4),dims=4)
    Δq = similar(data)
    for it=1:Nt
        Δq[:,:,:,it] = ψtoq(m, data[:,:,:,it])
    end
    qc = dropdims(mean(Δq, dims=4),dims=4)
    Δψ = similar(data)
    #get deviation from average
    for it=1:Nt
        Δψ[:,:,:,it] = data[:,:,:,it] - ψc
        Δq[:,:,:,it] = Δq[:,:,:,it] - qc
    end

elseif datasource=="qprime"

    qc = dropdims(mean(data, dims=4),dims=4)
    Δψ = similar(data)

    for it=1:Nt
        Δψ[:,:,:,it] = qprimetoψ(m, data[:,:,:,it])
    end
    ψc = dropdims(mean(Δψ, dims=4),dims=4)
    Δq = similar(data)
    #get deviation from average
    for it=1:Nt
        Δq[:,:,:,it] = data[:,:,:,it] - qc
        Δψ[:,:,:,it] = Δψ[:,:,:,it] - ψc
    end
else
    error("Unknown datasource, should be 'qprime' or 'sf'.")
end
heatmap(transform_grid(ψc, m)[1,:,:])

transform_grid(qc, m)
heatmap(transform_grid(qc, m)[1,:,:])
heatmap(transform_grid(Δq, m)[1,:,:,6])

#println("----")

#println(size(SH_dλ(Δψ[1,:,:,it], m).*SH_dϕ(Δq[1,:,:,it], m) - SH_dϕ(Δψ[1,:,:,it], m).*SH_dλ(Δq[1,:,:,it], m)))

#println("------")

#println(size((m.acosϕi ./ m.p.a)))
Jv = zeros(Float64, 3, m.p.L, m.p.M) # avg Jacobian variability
for it=1:Nt
    Jv[1,:,:] += J(Δψ[1,:,:,it], Δq[1,:,:,it], m)
    Jv[2,:,:] += J(Δψ[2,:,:,it], Δq[2,:,:,it], m)
    Jv[3,:,:] += J(Δψ[3,:,:,it], Δq[3,:,:,it], m)
end
Jv ./= Nt
Jv[1,:,:]

transform_grid(permutedims(cat( J(ψc[1,:,:], qc[1,:,:], m) + Jv[1,:,:] + D1(ψc, qc, m), J(ψc[2,:,:], qc[2,:,:], m) + Jv[2,:,:] + D2(ψc, qc, m), J(ψc[3,:,:], qc[3,:,:], m) + Jv[3,:,:] + D3(ψc, qc, m), dims=3),[3,1,2]),m)
"""
