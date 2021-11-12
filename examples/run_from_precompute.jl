# this file is an example how to run from a pre-computed model

using QG3, BenchmarkTools, DifferentialEquations, JLD2

# load forcing and model parameters
@load "data/t21-precomputed-S.jld2" S
@load "data/t21-precomputed-p.jld2" qg3ppars
@load "data/t21-precomputed-sf.jld2" ψ_0
@load "data/t21-precomputed-q.jld2" q_0

qg3p = QG3Model(qg3ppars)

# time step
DT = 2π/144
t_end = 500.

# problem definition with standard model from the library and solve
prob = ODEProblem(QG3.QG3MM_base, q_0, (0.,t_end), [qg3p, S])
sol = @time solve(prob, AB5(), dt=DT)

# PLOT OPtiON
using Plots
pyplot()

PLOT = true
if PLOT
        ilvl = 1  # choose lvl to plot here

        clims = (-1.1*maximum(abs.(ψ_0[ilvl,:,:])),1.1*maximum(abs.(ψ[ilvl,:,:,:]))) # get colormap maxima

        plot_times = 0:(t_end)/200:t_end  # choose timesteps to plot

        anim = @animate for (iit,it) ∈ enumerate(plot_times)
            sf_plot = transform_grid(qprimetoψ(qg3p, sol(it)),qg3p)
            heatmap(sf_plot[ilvl,:,:], c=:balance, title=string("time=",it,"   - ",it*qg3p.p.time_unit," d"), clims=clims)
        end
        gif(anim, "anim_fps20.gif", fps = 20)
 end
