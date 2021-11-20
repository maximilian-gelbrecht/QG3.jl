# QG3

This module provides a Julia implementation of the Marshall, Molteni QG3 Model. It runs on GPUs and is differentiable (tested only with Zygote).

The model is solved with a pseudo-spectral approach. Currently there is a naive implementation of Real Spherical Harmonics with transforms defined for a Gaussian Grid. There are bindings to FastTransforms.jl for a regular grid as well, however this is experimental and suffers from aliasing problems when integrating the equation.

There is a rudimentary documentation already available that is mostly all the doc strings from the individual functions. Unfortunately the PIK Gitlab offers no Gitlab Pages, so it is not hosted but hidden inside the `docs` folder.

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

Currently there are two different implementations, one that is optimised for CPU (the "2D version") and one that is optimised for GPU (the "3D version"). The GPU version can also run on CPU but not the other way around. Further explainations in the documentation.
