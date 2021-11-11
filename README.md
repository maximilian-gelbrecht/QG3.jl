# QG3

This code provides a Julia implementation of the Marshall, Molteni QG3 Model. It runs on GPUs and is differentiable (tested only with Zygote).

The model is solved with a pseudo-spectral approach. Currently there is a naive implemention of Real Spherical Harmonics with transforms defined for a Gaussian Grid. There are bindings to FastTransforms.jl for a regular grid as well, however this is experimental and suffers from aliasing problems when integrating the equation.

There is a rudimentary documentation already available that is mostly all the doc strings from the individual functions. Unfortunately the PIK Gitlab offers no Gitlab Pages, so it is not hosted but hidden inside the `docs` folder.

The `test/basic_test.jl` file also provides a basic example of how to run the model.
