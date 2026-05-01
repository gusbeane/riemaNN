#set math.equation(numbering: "(1)")
#set page(numbering: "1")

= Riemann Solvers with Neural Networks

Hydrodynamical systems must be solved numerically. This is typically done by mapping a discrete representation of the system onto a point-wise distribution that can be solved analytically for a short enough timestep. For example, taking the fluid's state within a cell to be constant and solving for the flux across the interface between two cells, which has an analytic solution. The basic idea here is to expand the scope of problems that can be solved analytically in order to improve both the accuracy and speed of simulations.

== Context

The basic problem in hydrodynamics is to solve for the time evolution of a fluid system specified by some initial conditions according to the Euler equations.#footnote[Technically the Navier-Stokes equations, but in most astrophysical systems the Euler equations are reasonably valid.] These are given in a compact form by (following Springel10),

$ partial_t bold(U) + nabla dot bold(F) = 0 $

where,

$ bold(U) &= vec(rho, rho bold(v), rho e) & wide wide bold(F) &= vec(rho bold(v), rho bold(v) bold(v)^T + P, (rho e + P) bold(v)) $ <defineUF>

and $rho$ is mass density, $bold(v)$ is velocity, $e = u + v^2/2$ is energy per unit mass and $u$ is thermal energy per unit mass, and $P$ is pressure. Note that these quantities are functions of space and time (e.g., $bold(U)=bold(U)(bold(x),t)$), but this will be implied henceforth. We have five equations and six unknowns ($rho$, $bold(v)$, $e$, and $P$), requiring a closure in the form of the "equation-of-state". For an ideal gas, this is given by $P=(gamma-1)rho u$ for some constant $gamma$ (e.g., $gamma=5/3$ for a monatomic gas, or $gamma=7/5$ for a diatomic gas).

== Numerical Pretext

=== Setup

Because of the $bold(v) bold(v)^T$ term in $bold(F)$, the Euler equations are nonlinear, and thus there are few exact analytic solutions. Therefore, numerical methods are necessary to make any progress on astrophysically relevant setups. The first step is to choose a discrete representation of the fluid. One can choose to have these specify the fluid state at a specific location according to some spatial discretization (Eulerian) or to have them represent a mass packet of fluid (Lagrangian).#footnote[In the Lagrangian case, one rearranges Euler's equations by an advective term (i.e., $partial_t -> D_t = partial_t + bold(v) dot nabla$).]

We focus exclusively on the Eulerian case. Furthermore, for simplicity, we assume a uniform Cartesian mesh as the discretization choice. On this mesh, our fluid state is stored as the cell average of each cell ${overline(bold(U))_i (t_j)}_(i=1..N)$ at discrete times $t_j$, given by

$ overline(bold(U))_i (t_j) = 1/V_i integral_("cell" i) bold(U)(bold(x), t_j) dif bold(x) $,

where $V_i$ is the volume of cell $i$. We'll assume a global uniform timestep, so that $t_(j+1)-t_j=Delta t$.

=== Integral Form of Euler's Equations

Our goal is clear: given the current state of the system ${overline(bold(U))_i (t_j)}_(i=1..N)$, predict the state at the next timestep of $overline(bold(U))_i (t_(j+1))$ for each cell $i$. For this, we rely on the integral form of Euler's equations. For clarity, operating in 1D, we integrate over the cell $i$ and across the timestep, i.e., over the region $[x_(i-1/2),x_(i+1/2)] times [t_j, t_(j+1)]$.#footnote[In higher dimensions, the flux integral becomes a surface integral over all the faces of a cell.] We've denoted $x_(i plus.minus 1/2)$ as the locations of the left/right faces of cell $i$. Then, the Euler equations become,

$ integral_(x_(i-1/2))^(x_(i+1/2)) [bold(U)(x,t_(j+1)) - bold(U)(x,t_j)] dif x + integral_(t_j)^(t_(j+1)) [bold(F)(x_(i+1/2), t) - bold(F)(x_(i-1/2), t)] dif t = 0 $.

Substituting our discrete values, and setting $V_i = Delta x equiv x_(i+1/2) - x_(i - 1/2)$, since we assume a uniform mesh, we have that,

$ overline(bold(U))_i (t_(j+1)) = overline(bold(U))_i (t_j) - 1/(Delta x) integral_(t_j)^(t_(j+1)) [bold(F)(x_(i+1/2), t) - bold(F)(x_(i-1/2), t)] dif t $ <discrete_euler>.

And so the problem of computing a cell's values at the next timestep is reduced to computing the integral of the fluxes at its interfaces over the timestep. Note that we haven't actually done any numerics yet --- this form of Euler's equation is valid for the true solution.

== Godunov's Method

=== Setup

Godunov's method is the canonical method of solving Euler's equation that is accurate to first-order. It is somewhat _ad hoc_. In fact, it is not guaranteed to converge to the correct solution --- *if* it converges, _then_ it is guaranteed to converge to the correct solution.#footnote[Convergence here is defined so that the cell averages converge to the average of the true solution in the limit that cell sizes go to zero. Inside baseball: convergence is *not* that the reconstruction converges point-wise to the true distribution. This is a good argument that the "true" discrete representation is just the piecewise constant cell averages.]

The basic idea is this: we can't solve Euler's equations for the true fluid distribution. So, instead, we will map our discrete representation (a list of values) onto a distribution that we _can_ solve analytically. In particular, we will evolve from $t_j$ to $t_(j+1)$ the piecewise-constant distribution $tilde(bold(U))$, defined as,

$ tilde(bold(U))(bold(x)) = overline(bold(U))_i quad forall bold(x) in "cell" i $
To be clear, ${overline(bold(U))_i}$ is a list of numbers while $tilde(bold(U))$ is a function well-defined across the domain.

This distribution has an exact solution that we _can_ solve analytically. Within a cell, far from the cell interfaces, there is no time evolution in the fluid state, provided the time step is not too large (implying a timestep criterion, mentioned below). At the interfaces between cells, the problem reduces to the 1D Riemann problem across the face, and so we can compute the integral of the flux across the timestep that is needed in @discrete_euler. All we need to know is the fluid quantities at the location of the cell interfaces at the time $Delta t$.

=== Riemann Problem

I'll write down the full solution to the Riemann problem at the end, but I'll sketch the structure of the solution and the salient numerical facts. This is all for an ideal gas with constant $gamma$.

First, the initial conditions are piecewise constant with 

$ bold(U)(x, 0) = cases(bold(U)_L & "if" x < 0, bold(U)_R & "if" x >= 0) $,

where $bold(U)_L$ and $bold(U)_R$ are constants.

The solution is self-similar in $xi = x/t$. It consists of three waves separating four regions:

$ L quad underbrace(|, "L-wave") quad L^* quad underbrace(|, "C-wave") quad R^* quad underbrace(|, "R-wave") quad R $

The L and R regions are simply the initial condition values. L-wave and R-wave can be either a shock (jump in $rho$, $v$, and $p$) or a rarefaction (smooth variation to C-wave), and C-wave is always a contact discontinuity (constant in $p$ and $bold(v)$, jump in $rho$). The pressure within L\* and R\* is always constant and denote $p^*$.#footnote[The speed of the L-wave and R-wave is what determines the timestep criterion, since we can't have them propagate for long enough that they start interacting with the cell faces in complicated ways. So typically one enforces that the timestep times the fastest wave is less than half the cell size. Technically in 2D and 3D the waves from perpendicular faces will intersect and interact immediately, but this introduces error at second order.]

The structure of the L\* and R\* regions are determined by L-wave and R-wave, respectively --- constant in the case of a shock and a smooth variation in the case of a rarefaction. If $p^* > p_L$, then L-wave is a shock, and if $p^* < p_L$ then L-wave is a rarefaction. Similarly for R-wave and $p_R$.

There are two important things to take from the Riemann problem. First, we only need the value at $xi=0$. This corresponds to $bold(F)(x_(i plus.minus 1/2), t)$ in @discrete_euler. Second, the fluid state within the star-region (between L-wave and R-wave) is determined entirely by $p^*$. Once $p^*$ has been computed, the other star-state quantities can be computed straightforwardly.

There is no closed-form solution to find $p^*$. Instead, we write down an equation $f(p)$ such that $f(p^*)=0$, and then find the solution iteratively. This is given later.

== Source Terms

=== Setup

Earlier, when we introduced Euler's equations, we ignored source terms which capture additional physics processes, such as self-gravity, external gravity, heating/cooling, mass injection (e.g., stellar winds), etc. Here, we will consider only gravity. The modification to Euler's equation is then,

$ partial_t bold(U) + nabla dot bold(F) &= bold(S) & wide wide bold(S) &= vec(0, -rho nabla Phi, -rho bold(v) dot nabla Phi) $,

where $Phi$ is the gravitational potential which can be solved in the typical ways.

Unfortunately, there is no analytic solution to the general Riemann problem (i.e., in the case with source terms), so we can't just compute the fluxes by incorporating an additional term. The solution is to use operator splitting.

=== Operation Splitting

In our case, the basic idea behind operator splitting is to alternate between a source term like gravity and hydro. So, advance the system for a hydro step ignoring gravity, then apply gravity for a timestep ignoring hydro, and so on.

Suppose we have a very simple ODE for $x(t)$ given by 
$ x' = f(x) + g(x) $,
where $f(x)=a x$ and $g(x) = b x$ and initial data $x(0)=x_0$. In this case, the solution at some time $t$ in the future is trivially given by,
$ x = x_0 e^((a+b)t) $
These operations can be split, i.e.,
$ x = e^(b t) e^(a t) x_0 $.

Because this system is linear, the solution is exact, but for nonlinear systems the splitting technique works and still gives first-order accuracy. So, in general, if we have
$ x' = f(x) + g(x) $
then the solution can be approximated as
$ x(t_(i+1)) = e^(g Delta t)e^(f Delta t)x(t_(i)) $
which is accurate to $cal(O)(Delta t)$. Here, $e^(f Delta t)$ is the solution to the equation $x' = f(x)$, and same for $g$.

The error introduced by this can be drawn from the Baker-Campbell-Hausdorff, which states that
$ e^(f t) e^(g t) = e^((f+g)t + 1/2 [f,g]t^2 + 1/12 ([f,[f,g]] + [g,[g,f]])t^3 + cal(O)(t^4)) $ <BCH>
where $[f, g](x) = (partial g)/(partial x) f(x) - (partial f) / (partial x) g(x)$ is the Lie bracket. In first-order operator splitting, the leading order error term is $1/2 [f,g]t^2$. In the linear case, we had that $[f,g]=0$, and so the split solution is exact.

Strang splitting extends this to second order, $cal(O)(Delta t^2)$ by changing order of operations:
$ x(t_(i+1)) = e^(f 1/2 Delta t)e^(g Delta t)e^(f 1/2 Delta t)x(t_(i)) $
The leading order error is now the $t^3$ term in @BCH.

So, in our case $f$ and $g$ are given by the fluxes, $nabla dot bold(F)$ and the source terms $bold(S)$.#footnote[Strang splitting can, of course, be extended to arbitrary number of functions, and so multiple processes accounted for in the source terms can be further split.] Strang splitting is usually good enough, but in systems in hydrostatic equilibrium, the solution can drift because the errors from the gravity solver and the hydro solver don't cancel to machine precision.

Another way to think about this is that gravity and hydro don't commute -- i.e., $[nabla dot bold(F), bold(S)] != 0$. In cases with strong shocks or strong gravity, these higher-order cutoff terms can become significant. (I think the way to think about this is that Strang splitting actually introduces another timestep criterion in that the $cal(O)(t^3)$ term in @BCH is small, which could be more restrictive than the typical hydro timestep criterion. I don't think this is ever done in practice.)

== Neural Networks

=== Ideal Gas

In the way current hydro codes based on the Godunov scheme are setup, the crucial step comes down to solving for the flux across each face at each timestep. Returning to the ideal gas Riemann problem, this can be thought of in terms of a neural network that maps the left and right states to the star-state pressure:
$ cal(N): vec(bold(U)_L, bold(U)_R) -> p^* $
where, recall that in the ideal gas case all fluid quantities in the star-state can be derived from $p^*$. This includes the state at $xi=x/t=0$, which is all that is needed for the flux.

=== Flux Time Evolution

In the ideal gas case, because of the self-similarity of the solution, the fluid state at the cell interface ($xi=0$) is constant with time. But this is not true in general, and so one does truly need to know the full time evolution of $F$ in @discrete_euler. In this case, I think we can just expand $F$ as a Taylor series and predict the coefficients, so that the flux at any arbitrary time in the future can be predicted. E.g.,
$ cal(N): vec(bold(U)_L, bold(U)_R) -> vec((rho_0, rho_1, rho_2, ...), (bold(v)_0, bold(v)_1, bold(v)_2, ...), (u_0, u_1, u_2, ...)) $
where $rho(t) approx rho_0 + rho_1 t + 1/2 rho_2 t^2 + ...$, and similarly for $bold(v)$ and $u$.

One case in which the flux is not constant in time at cell interfaces comes from source terms, e.g., in the presence of gravity. In the case of gravity, there is a perturbative solution to the Riemann problem to arbitrary order, but for arbitrary source terms this will not be the case. My thinking is to eventually have the machinery setup to run high-fidelity 1D simulations (high resolution, 5th order reconstruction, exact Riemann solvers, etc) to generate massive training sets where the Riemann problem has been solved to machine precision, and then use these to train a neural network to predict the fluxes.

=== General Equation of State

There was of course the original motivation for doing this, which was to solve the Riemann problem with general equations of state. Here the approximate Riemann solvers can fail in some regimes, and exact solvers are far too expensive, and so a neural network accelerated solver could be a solution.

=== Towards Third-Order
Another idea is that in 2D and 3D, the basic assumption of Godunov schemes break down. In 1D, you restrict the timestep such that the wave coming from the right does not interact with the wave coming from the left (or that their interaction does not propagate to the cell boundaries). However, in 2D, waves coming from perpendicular faces (e.g., the $+x$ and $+y$ face) will interact immediately, and can, in general, backreact and return to the cell faces immediately. This is generally ignored, which is okay becuase it introduces errors only at second error, and most schemes are second-order to begin with. But one can train a neural network now on the full 2D/3D setup to account for this interaction. This dramatically increases the input dimensions (you now need to account for 9 input cell states instead of 2 in 2D, and 27 input states in 3D), so I don't know if it would work. But this, along with avoiding Strang splitting, could be a path to efficient third order schemes.

#pagebreak()
== Addendum: Exact Solution to the 1D Riemann Problem (arbitrary $gamma$)

I did not write this section. This is AI slop.

=== Setup

Initial states $(rho_L, v_L, p_L)$ and $(rho_R, v_R, p_R)$ separated at $x = 0$. Sound speed $c_K = sqrt(gamma p_K \/ rho_K)$. The solution is self-similar in $xi = x\/t$.

Define the following shorthand constants:

$ alpha = (gamma - 1) / (2 gamma), quad beta = (gamma - 1) / (gamma + 1), quad mu = (gamma - 1) / 2 $

=== Step 1: Find the Star-State Pressure $p^*$

Solve $f(p^*) = 0$ iteratively, where

$ f(p) = f_L (p) + f_R (p) + (v_R - v_L) $

For each side $K in {L, R}$:

_Shock_ ($p > p_K$):

$ f_K (p) = (p - p_K) sqrt(A_K / (p + B_K)) $

where $A_K = 2 \/ ((gamma+1) rho_K)$ and $B_K = (gamma - 1)\/(gamma + 1) p_K = beta p_K$.

_Rarefaction_ ($p <= p_K$):

$ f_K (p) = (2 c_K) / (gamma - 1) [(p / p_K)^alpha - 1] = c_K / mu [(p / p_K)^alpha - 1] $

=== Step 2: Star-State Velocity and Densities

$ v^* = 1/2 (v_L + v_R) + 1/2 [f_R (p^*) - f_L (p^*)] $

_Shock_ ($p^* > p_K$):

$ rho^*_K = rho_K (p^* \/ p_K + beta) / (1 + beta p^* \/ p_K) = rho_K ((gamma+1) p^* + (gamma-1) p_K) / ((gamma-1) p^* + (gamma+1) p_K) $

_Rarefaction_ ($p^* <= p_K$):

$ rho^*_K = rho_K (p^* / p_K)^(1\/gamma) $

=== Step 3: Wave Speeds

_Left shock_ ($p^* > p_L$):

$ S_L = v_L - c_L sqrt(1 + (gamma+1)/(2 gamma) (p^* \/ p_L - 1)) $

_Left rarefaction_ ($p^* <= p_L$):

$ S_"head" = v_L - c_L, quad S_"tail" = v^* - c^*_L $

where $c^*_L = c_L (p^* \/ p_L)^alpha$.

_Right shock_ ($p^* > p_R$):

$ S_R = v_R + c_R sqrt(1 + (gamma+1)/(2 gamma) (p^* \/ p_R - 1)) $

_Right rarefaction_ ($p^* <= p_R$):

$ S_"head" = v_R + c_R, quad S_"tail" = v^* + c^*_R $

where $c^*_R = c_R (p^* \/ p_R)^alpha$.

_Contact discontinuity:_ $S_"contact" = v^*$.

=== Step 4: Sample the Solution at $xi = x\/t$

_Left rarefaction fan_ ($S_"head" <= xi <= S_"tail"$):

$ v = 2/(gamma+1) (xi + c_L + mu v_L), quad c = 2/(gamma+1) (c_L + (v_L - xi)/2) $

$ rho = rho_L (c / c_L)^(2\/(gamma-1)), quad p = p_L (c / c_L)^(2 gamma\/(gamma-1)) $

_Right rarefaction fan_ ($S_"tail" <= xi <= S_"head"$):

$ v = 2/(gamma+1) (xi - c_R + mu v_R), quad c = 2/(gamma+1) (c_R + (v_R - xi)/2) $

$ rho = rho_R (c / c_R)^(2\/(gamma-1)), quad p = p_R (c / c_R)^(2 gamma\/(gamma-1)) $

