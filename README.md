![QG3 Logo](logo.png)

# QG3.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maximilian-gelbrecht.github.io/QG3.jl/dev/)
[![Build Status](https://github.com/maximilian-gelbrecht/QG3.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/maximilian-gelbrecht/QG3.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/807073084.svg)](https://doi.org/10.5281/zenodo.14547915)

This package provides a Julia implementation of the [Marshall, Molteni Quasigeostrophic Atmsopheric Model in three layers](https://journals.ametsoc.org/view/journals/atsc/50/12/1520-0469_1993_050_1792_taduop_2_0_co_2.xml). It runs on CPU or CUDA GPUs and is differentiable via Zygote.jl.

The model is solved with a pseudo-spectral approach on a gaussian grid with approatiate spherical harmonics transforms defined. Example scripts are provided in the `examples` folder.

Install e.g. via `]add https://github.com/maximilian-gelbrecht/QG3.jl.git` and test the installation with `]test QG3`

The repository includes pre-computed forcing and initial conditions on a T21 to run the model but no proper dataset to re-compute those in order to save space. The example folder also includes the necessary scripts to pre-compute those for other grids and datasets. 

## The Model

Details about the model can be read up in ["Towards a Dynamical Understanding of Planetary-Scale Flow Regimes", Marshall, Molteni, Journal of Atmsopheric Sciences, 1993](https://journals.ametsoc.org/view/journals/atsc/50/12/1520-0469_1993_050_1792_taduop_2_0_co_2.xml)

Its governing equation for the quasigeostrophic vorticity $`q_i`$ in three equipressure levels (200, 500, 800 hPa) are given be

```math
\dot{q_i} = -J(\psi_i, q_i) - D(\psi_i, q_i) + S \\
\vec{q} = T_{\psi q} \vec{\psi}
```
where the voriticy $`q`$ and streamfunction $`\psi`$ are related by a linear operator (comprising the Laplacian and temperature relaxation), $`J`$ is the Jacobian / advection term, $`D`$ the dissipation and $`S`$ a forcing computed from data.

Currently there are two different implementations, one that is optimised for CPU (the "2D version") and one that is optimised for GPU (the "3D version"). The GPU version can also run on CPU, albeit a bit slower, but not the other way around. Further explainations in the documentation.

## The Future 

It's unlikely that this model will see a lot of further development. But key ideas and lessons learned from this project are applied to [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl). We are currently working on a AD and GPU enabled version of this much more complex atmospheric model. 

## Cite us 

If you are using this model for any publications or other works, please cite us. 

[![DOI](https://zenodo.org/badge/807073084.svg)](https://doi.org/10.5281/zenodo.14547915)
