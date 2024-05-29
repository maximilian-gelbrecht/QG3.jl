# QG3.jl

This package provides a Julia implementation of the [Marshall, Molteni Quasigeostrophic Atmsopheric Model in three layers](https://journals.ametsoc.org/view/journals/atsc/50/12/1520-0469_1993_050_1792_taduop_2_0_co_2.xml). It runs on CPU or CUDA GPUs and is differentiable via Zygote.jl.

The model is solved with a pseudo-spectral approach on a gaussian grid with approatiate spherical harmonics transforms defined. Example scripts are provided in the `examples` folder.

Install e.g. via `]add https://github.com/maximilian-gelbrecht/QG3.jl.git` and test the installation with `]test QG3`

The repository includes pre-computed forcing and initial conditions on a T21 to run the model but no proper dataset to compute re-compute those in order to save space. The example folder also includes the necessary scripts to pre-copmute those for other grids and datasets. 

## The Model

Details about the model can be read up in "Towards a Dynamical Understanding of Planetary-Scale Flow Regimes", Marshall, Molteni, Journal of Atmsopheric Sciences, 1992.

Its governing equation for the quasigeostrophic vorticity $`q_i`$ in three equipressure levels (200, 500, 800 hPa) are given be

```math
\dot{q_i} = -J(\psi_i, q_i) - D(\psi_i, q_i) + S \\
\vec{q} = T_{\psi q} \vec{\psi}
```
where the voriticy $`q`$ and streamfunction $`\psi`$ are related by a linear operator (comprising the Laplacian and temperature relaxation), $`J`$ is the Jacobian / advection term, $`D`$ the dissipation and $`S`$ a forcing computed from data.

## Solvers 

This packages includes all function necessary to define the governing equation of the QG3 Model. It does however, not include the differential equations solvers, for those we rely on `DifferentialEquations.jl`, i.e. `OrdinaryDiffEq.jl`. In the following, we will therefore briefly talk about how to define the right hand side necessary to define for those solvers, before going into a little more detail on the transforms and derivatives used throughout the model. 

## Initialize a model 

The model parameter are hold as [`QG3ModelParameters`](@ref): 

```@docs; canonical=false 
QG3ModelParameters
```

For convenience, a pre-computed setup with forcing `S`, parameters `qg3_pars` and initial conditions `ψ_0`, `q_0` can be loaded via 

```julia 
S, qg3_pars, ψ_0, q_0 = QG3.load_precomputed_data(GPU=false)
```

Alternatively, the `pre-compute.jl` script in the examples folder demonstrates how to compute those from other datasets and for other resolutions. 

Then, the model can be pre-computed with the help of [`QG3Model`](@ref)

```julia
qg3 = QG3Model(qg3_pars)
```

or on GPU 

```julia 
qg3 = CUDA.@allowscalar QG3Model(qg3_pars) # on GPU the pre-computation need scalar indexing
```

These struct also hold the grid information `qg3.g` that comprises all transforms and derivatives. 

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

This right hand side calls the transform from potential vorticity to the streamfunction [`qprimetoψ`](@ref), the advection operator [`J`](@ref), the dissipation operator [`D`](@ref) and the pre-computed forcing `S`. 

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

### Model Units 

This implemenation uses a unit system, so that the Earth's radius and angular velocity equals

```math 
\begin{aligned}
R = 1
2\Omega = 1 
\end{aligned}
```

, from these follow the units for all variables. The parameter struct holds those as `qg3_pars.time_unit`, `qg3_pars.distance_unit`, `qg3_pars.ψ_unit` and `qg3_pars.q_unit`. 

### Transforms

Throughout the model real spherical harmonics are used. The transform is implemented with a FFT and a Gauss-Legendre transform that is solved by pre-computing the associated Legendre polynomials and then executing the appropiate matrix multiplication for the transform. This ensures both differentiability and GPU compatability of the transforms. Gradients of the spherical harmonics transforms are checked against finite difference differnetiation in the tests.  

Transforms are called with [`transform_SH`](@ref) and [`transform_grid`](@ref), e.g. as in the following: 

```julia 
ψ_0_grid = transform_grid(ψ_0, qg3)
ψ_0_SH = transform_SH(ψ_0_grid, qg3)
```

SHs coefficients on CPU are handled in the matrix convention that FastTransforms.jl uses: columns by m-value: 0, -1, 1, -2, 2, ..., rows l in ascending order. This is implemented initially during development to ensure compatability with FastTransforms.jl, unterfortunately however, using FastTransforms.jl isn't working due to aliasing problems. On GPU, we store the coefficients differentialy  naive SH transform definately not the fasted way of storing the coefficients as an additonal allocating reordering needs to be done for every transform. Therefore the coefficient matrix convention is different on GPU, where the columns are ordered 0, 1, 2, .... l_max, 0, -1, -2, -3, .. . 

### Derivates 

Whereas zonal derivatives and the Laplacian are computed in the SH domain, meridional derivatives are computed pseudo-spectrally with pre-computed values of the derivatives of the associated Legendre polynomials. 

All derivatives follow a common naming scheme: 

```julia 
SHtoSH_dλ(ψ_0, qg3) # zonal derivative SH -> SH 
SHtoGrid_dλ(ψ_0, qg3) # zonal derivative SH -> Grid
SHtoGrid_dμ(ψ_0, qg3) # meridional (μ = sin(latitude)) derivative SH -> Grid 
SHtoGrid_dθ(ψ_0, qg3) # meridional (colatitude) derivative SH -> Grid 
Δ(ψ_0, qg3) # Laplacian
```

### Transforms and Derivates for Machine Learning 

Outside of the QG3 Model itself the transforms and derivative can also be used for ML tasks. For this purpose the constructors also allow to define them for more horizonatal levels and also for an additional (batch) dimension. The batch dimensions is added as a forth dimension, thus defining transforms and derivaties for fields `(lvl x N_lat x N_lon x N_batch)`. 

Either, the transforms and derivatives are initialized directly with the approatite `N_level` and `N_batch` input arguments via 

```@docs; canonical=false 
GaussianGridtoSHTransform
SHtoGaussianGridTransform
Derivative_dλ
GaussianGrid_dμ
Laplacian
```

or a they can be constructed all together via 

```@docs; canonical=false 
grid 
```