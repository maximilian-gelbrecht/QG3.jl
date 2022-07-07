# this is a basic run from netcdf files, you have to make sure that the data is there. it is not present in the repository where you'll only find example initial conditions and a pre-computed forcing in order to save space in the repository / package.

using QG3, NetCDF, CFTime, Dates, BenchmarkTools, DifferentialEquations

# first we import the data (streamfunction), land sea mask, orography etc
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

# time step
DT = T(2π/144)
t_end = T(500.)

# problem definition with standard model from the library and solve
prob = ODEProblem(QG3.QG3MM_base, q_0, (T(0.),t_end), (qg3p, S))
#sol = @time solve(prob, AB5(), dt=DT)
sol = @time solve(prob, Tsit5(), dt=DT)



# PLOT OPtiON
using Plots

PLOT = true
if PLOT
        ilvl = 1  # choose lvl to plot here

        clims = (-1.1*maximum(abs.(ψ[ilvl,:,:,:])),1.1*maximum(abs.(ψ[ilvl,:,:,:]))) # get colormap maxima

        plot_times = 0:(t_end)/500:500.  # choose timesteps to plot

        anim = @animate for (iit,it) ∈ enumerate(plot_times)
            sf_plot = transform_grid(qprimetoψ(qg3p, sol(Float32(it))),qg3p)
            heatmap(sf_plot[ilvl,:,:], c=:balance, title=string("time=",it,"   - ",it*qg3p.p.time_unit," d"), clims=clims)
        end
        gif(anim, "anim_fps20.gif", fps = 20)
 end

"""
 if PLOT
        ilvl = 1  # choose lvl to plot here

        clims = (-1.1*maximum(abs.(ψ[ilvl,:,:,:])),1.1*maximum(abs.(ψ[ilvl,:,:,:]))) # get colormap maxima

        plot_times = 0:(t_end)/200:t_end  # choose timesteps to plot

        anim = @animate for (iit,it) ∈ enumerate(plot_times)
            heatmap(ψ[ilvl,:,:,iit], c=:balance, title=string("time=",it,"   - ",it*qg3p.p.time_unit," d"), clims=clims)
        end
        gif(anim, "data_fps20.gif", fps = 20)
 end
"""