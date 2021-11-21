# QG3 Model

This module provides a Julia implementation of the Marshall, Molteni QG3 Model. It runs on GPUs and is differentiable (tested only with Zygote).

The model is solved with a pseudo-spectral approach. Currently there is a naive implemention of Real Spherical Harmonics with transforms defined for a Gaussian Grid. There are bindings to FastTransforms.jl for a regular grid as well, however this is experimental and suffers from aliasing problems when integrating the equation.

So far this documentation is not yet fully fledged out. It just provides the docstrings and some very basic how-to.

There are example scripts in the `examples` folder.

Install e.g. via `]add https://gitlab.pik-potsdam.de/maxgelbr/qg3.jl.git` and test the installation with `]test QG3`

The repository includes pre-computed forcing and initial conditions to run the model but no proper dataset in order to save space.

## The Model

Details about the model can be read up in "Towards a Dynamical Understanding of Planetary-Scale Flow Regimes", Marshall, Molteni, Journal of Atmsopheric Sciences, 1992.

Its governing equation for the quasigeostrophic vorticity $`q_i`$ in three equipressure levels (200, 500, 800 hPa) are given be

```math
\dot{q_i} = -J(\psi_i, q_i) - D(\psi_i, q_i) + S \\
\vec{q} = T_{\psi q} \vec{\psi}
```
where the voriticy $`q`$ and streamfunction $`\psi`$ are related by a linear operator (comprising the Laplacian and temperature relaxation), $`J`$ is the Jacobian / advection term, $`D`$ the dissipation and $`S`$ a forcing computed from data.

## GPU and CPU

Currently there are two different implementations, one that is optimised for CPU (the "2D version") and one that is optimised for GPU (the "3D version"). The GPU version can also run on CPU but not the other way around.

### GPU Version

The GPU version is fully vectorized, without scalar indexing. All needed functions work on the 3d (level, lat, lon) / (level, il, m) field. The full problem definition thus is just simply

```julia
function QG3MM_gpu(q, m, t)
    p, S = m # parameters, forcing vector

    ψ = qprimetoψ(p, q)
    return - J(ψ, q, p) - D(ψ, q, p) + S
end
```

### CPU Version

The CPU version works on 2d fields (lat, lon) / (il, m), on CPUs this is currently a little faster than using the GPU/3D version on CPU. In other words: the GPU version could probably still be optimized further. The problem definition follows as

```julia
function QG3MM_base(q, m, t)
    p, S = m # parameters, forcing vector

    ψ = qprimetoψ(p, q)
    return cat(
    reshape(- J(ψ[1,:,:], q[1,:,:], p) .- D1(ψ, q, p) .+ S[1,:,:], (1, p.p.L, p.p.M)),
    reshape(- J(ψ[2,:,:], q[2,:,:], p) .- D2(ψ, q, p) .+ S[2,:,:], (1, p.p.L, p.p.M)),
    reshape(- J3(ψ[3,:,:], q[3,:,:], p) .- D3(ψ, q, p) .+ S[3,:,:], (1, p.p.L, p.p.M)),
    dims=1)
end
```
