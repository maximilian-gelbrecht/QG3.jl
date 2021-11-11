# this test just checks if the model successfully compiles and integrates for short times in its most basic version

using QG3, NetCDF, CFTime, Dates, BenchmarkTools, DifferentialEquations

begin
        DIR = "../data/"
        NAME = "ERA5-sf-t21q.nc"
        LSNAME = "land-t21.nc"
        ORONAME = "oro-t21.nc"

        LATNAME = "lat"
        LONNAME = "lon"

        lats = deg2rad.(Float64.(ncread(string(DIR,NAME),LATNAME)))
        lat_inds = 1:size(lats,1)

        ψ = ncread(string(DIR,NAME),"atmosphere_horizontal_streamfunction")[:,:,:,:]

        lvl = ncread(string(DIR,NAME),"level")
        lats = deg2rad.(Float64.(ncread(string(DIR,NAME),LATNAME)))[lat_inds]
        lons = deg2rad.(Float64.(ncread(string(DIR,NAME),LONNAME)))

        times = CFTime.timedecode( ncread(string(DIR,NAME),"time"),ncgetatt(string(DIR,NAME),"time","units"))

        summer_ind = [month(t) ∈ [6,7,8] for t ∈ times]
        winter_ind = [month(t) ∈ [12,1,2] for t ∈ times]

        LS = Float64.(permutedims(ncread(string(DIR,LSNAME),"var172")[:,:,1],[2,1]))[lat_inds,:]
        # Land see mask, on the same grid as lats and lons

        h = (Float64.(permutedims(ncread(string(DIR,ORONAME),"z")[:,:,1],[2,1]))[lat_inds,:] .* ncgetatt(string(DIR,ORONAME), "z", "scale_factor")) .+ ncgetatt(string(DIR,ORONAME),"z","add_offset")
        # orography, array on the same grid as lats and lons

        LEVELS = [200, 500, 800]

        ψ = togpu(ψ[:,:,level_index(LEVELS,lvl),:])
        ψ = permutedims(ψ, [3,2,1,4]) # level, lat, lon,
        ψ = Float64.(ψ[:,lat_inds,:,:])


        gridtype="gaussian"
end

L = 22 # T21 grid
qg3ppars = QG3ModelParameters(L, lats, lons, LS, h)

ψ = ψ ./ qg3ppars.ψ_unit

qg3p = QG3Model(qg3ppars)

ψ_SH = transform_SH(ψ[:,:,:,1:10000], qg3p)

ψ_0 = ψ_SH[:,:,:,1]
q_0 = QG3.ψtoqprime(qg3p, ψ_0)

S = @time QG3.compute_S_Roads(ψ_SH[:,:,:,1:10000], qg3p)

DT = 2π/144

t_end = 200.

prob = ODEProblem(QG3.QG3MM_base, q_0, (0.,t_end), [qg3p, S])
sol = @time solve(prob, AB5(), dt=DT)

if sol.retcode==:Success
        return true
else
        return false;
end
