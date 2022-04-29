using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPModelTools
using POMCPOW
using POMDPSimulators
using ParticleFilters
using POMDPPolicies

## Playing around with the POMDP

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP()

# Plot the belief and the ground truth state
b = initialstate(pomdp)
s0 = rand(b)

plot()
for p in rand(b, 100)
    plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
end
plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")

# Here are the possible actions
as = actions(pomdp)

# We can call gen to start injecting
sp1, o1, r1 = gen(pomdp, s0, (:inject, 0.1))

# The observation is initially empty
@assert o1 == []

# We can then call gen to place an observation well at x=0.6
sp2, o2, r2 = gen(pomdp, sp, (:observe, 0.6))

# We can see the observation well and the injected CO2 by rendering
render(pomdp, sp2)

# The resulting observations gives a noisy estimate of the amount of CO2 below
o

## Solving the POMDP

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=100)
planner = solve(solver, pomdp)

# Run two different solvers
simulate(RolloutSimulator(), pomdp, RandomPolicy(pomdp), BootstrapFilter(pomdp, 100))
simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100))