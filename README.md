# Physics-Informed Neural Network Collection 

This repository contains **three Physics-Informed Neural Network (PINN)** examples implemented in PyTorch:

1. **Two-Electron Schrödinger Equation** (`two_electron_one_dimensional_schrodinger_differential_nn.ipynb`)  
   A PINN solving the 1D, two-particle Schrödinger equation with soft-Coulomb interactions. A single network models the **joint wavefunction** ψ(x₁, x₂), enforcing Dirichlet boundary conditions and normalization, and includes E (energy) as a trainable parameter.

2. **Heat Equation PINN** (`heat_equation.ipynb`)  
   Demonstrates a PINN solving the 1D heat/diffusion equation (∂u/∂t = α ∂²u/∂x²). This notebook uses PyTorch autograd to compute spatial and temporal derivatives, enforces Dirichlet boundaries and initial conditions, and compares the neural solution with the analytic form.

3. **Manual-Derivative ODE PINN** (`absolute_univariable_differential.ipynb`)  
   A simple ODE solver for df/dx = f(x), with the exact solution f(x) = C eˣ. Instead of using autograd, it performs **manual chain-rule** computation of derivatives through each network layer. Ideal for pedagogy and understanding how derivatives propagate via neural architectures.

---

## Next step
- Currently, developing solutions for the 3d navier stokes equations (numerical)
- We will need to fit the equation to a unit hypershere and then experiment with different constraints
- We can also implement korobov transformations on the features for more robust convergence

