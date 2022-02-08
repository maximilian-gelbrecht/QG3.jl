# this is an example how to pre-compute the model (and not run it) from netcdf files, you have to make sure that the data is there.

using QG3, NetCDF, CFTime, Dates, BenchmarkTools, DifferentialEquations, JLD2

# first we import the data (streamfunction), land sea mask, orography etc

# precision type (use Float32 or Float64)
T = Float32

begin
        DIR = "data/"
        NAME = "ERA5-sf-t21q.nc"
        LSNAME = "land-t21.nc"
        ORONAME = "oro-t21.nc"

        LATNAME = "lat"
        LONNAME = "lon"

        lats = deg2rad.(T.(ncread(string(DIR,NAME),LATNAME)))
        lat_inds = 1:size(lats,1)

        ψ = ncread(string(DIR,NAME),"atmosphere_horizontal_streamfunction")[:,:,:,:]

        lvl = ncread(string(DIR,NAME),"level")
        lats = deg2rad.(T.(ncread(string(DIR,NAME),LATNAME)))[lat_inds]
        lons = deg2rad.(T.(ncread(string(DIR,NAME),LONNAME)))

        times = CFTime.timedecode( ncread(string(DIR,NAME),"time"),ncgetatt(string(DIR,NAME),"time","units"))

        summer_ind = [month(t) ∈ [6,7,8] for t ∈ times]
        winter_ind = [month(t) ∈ [12,1,2] for t ∈ times]

        LS = T.(permutedims(ncread(string(DIR,LSNAME),"var172")[:,:,1],[2,1]))[lat_inds,:]
        # Land see mask, on the same grid as lats and lons

        h = (T.(permutedims(ncread(string(DIR,ORONAME),"z")[:,:,1],[2,1]))[lat_inds,:] .* T.(ncgetatt(string(DIR,ORONAME), "z", "scale_factor"))) .+ T.(ncgetatt(string(DIR,ORONAME),"z","add_offset"))
        # orography, array on the same grid as lats and lons

        LEVELS = [200, 500, 800]

        ψ = togpu(ψ[:,:,level_index(LEVELS,lvl),:])
        ψ = permutedims(ψ, [3,2,1,4]) # level, lat, lon,
        ψ = T.(ψ[:,lat_inds,:,:])

        gridtype="gaussian"
end

L = 22 # T21 grid, truncate with l_max = 21

# pre-compute the model and normalize the data
qg3ppars = QG3ModelParameters(L, lats, lons, LS, h)

ψ = ψ ./ qg3ppars.ψ_unit

qg3p = QG3Model(qg3ppars)

# stream function data in spherical domain
ψ_SH = transform_SH(ψ, qg3p)

# initial conditions for streamfunction and vorticity
ψ_0 = ψ_SH[:,:,:,1]
q_0 = QG3.ψtoqprime(qg3p, ψ_0)

# compute the forcing from winter data
S = @time QG3.compute_S_Roads(ψ_SH[:,:,:,winter_ind], qg3p)

@save "data/t21-precomputed-S.jld2" S
@save "data/t21-precomputed-p.jld2" qg3ppars
@save "data/t21-precomputed-sf.jld2" ψ_0
@save "data/t21-precomputed-q.jld2" q_0
