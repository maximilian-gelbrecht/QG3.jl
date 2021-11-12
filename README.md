# QG3

This code provides a Julia implementation of the Marshall, Molteni QG3 Model. It runs on GPUs and is differentiable (tested only with Zygote).

The model is solved with a pseudo-spectral approach. Currently there is a naive implementation of Real Spherical Harmonics with transforms defined for a Gaussian Grid. There are bindings to FastTransforms.jl for a regular grid as well, however this is experimental and suffers from aliasing problems when integrating the equation.

There is a rudimentary documentation already available that is mostly all the doc strings from the individual functions. Unfortunately the PIK Gitlab offers no Gitlab Pages, so it is not hosted but hidden inside the `docs` folder.

There are example scripts in the `examples` folder.

Install e.g. via `]add https://gitlab.pik-potsdam.de/maxgelbr/qg3.jl.git` and test the installation with `]test QG3`

The repository includes pre-computed forcing and initial conditions to run the model but no proper dataset in order to save space.
