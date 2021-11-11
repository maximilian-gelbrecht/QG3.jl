# QG3 Model

This code provides a Julia implementation of the Marshall, Molteni QG3 Model. It runs on GPUs and is differentiable (tested only with Zygote).

The model is solved with a pseudo-spectral approach. Currently there is a naive implemention of Real Spherical Harmonics with transforms defined for a Gaussian Grid. There are bindings to FastTransforms.jl for a regular grid as well, however this is experimental and suffers from aliasing problems when integrating the equation.

So far this documentation is not yet fully fledged out. It just provides the docstrings and some very basic how-to.

Check out the `basic_test.jl` for a simple minimal example of the running the QG3 system. 
