using Revise
using Pkg
# Pkg.activate()
# Pkg.develop(path=".")
using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPModelTools
using POMCPOW
using POMDPSimulators
using ParticleFilters
using POMDPPolicies
using D3Trees
using Distances
using Plots



## Playing around with the POMDP

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP()

# Plot the belief and the ground truth state
b = initialstate(pomdp)
s0 = rand(b)

# plot()
# for p in rand(b, 100)
#     plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
# end
# plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")

# Here are the possible actions
as = actions(pomdp)

# We can call gen to start injecting
sp1, o1, r1 = gen(pomdp, s0, (:inject, 0.1))


up = SubsurfaceUpdater(pomdp)
b2 = update(up, deepcopy(b), (:inject, 0.1), o1)

# The observation is initially empty
@assert o1 == []

# We can then call gen to place an observation well at x=0.6
sp2, o2, r2 = gen(pomdp, sp1, (:observe, 0.6))
b3 = update(up, deepcopy(b2), (:observe, 0.6), o2)

sp3, o3, r3 = gen(pomdp, sp2, (:observe, 0.65))
b4 = update(up, deepcopy(b3), (:observe, 0.65), o3)

sp4, o4, r4 = gen(pomdp, sp3, (:observe, 0.45))
b5 = update(up, deepcopy(b4), (:observe, 0.45), o4)


## Plotting

plot()
for p in rand(b5, 100)
    plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
end
plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")






# We can see the observation well and the injected CO2 by rendering
render(pomdp, sp2, "")

# The resulting observations gives a noisy estimate of the amount of CO2 below
o2

## Solving the POMDP







# Setup and run the solver
solver = POMCPOWSolver(tree_queries=100, criterion=MaxUCB(20.0), tree_in_info=true)
planner = solve(solver, pomdp)

# Run two different solvers

hist = simulate(HistoryRecorder(), pomdp, planner, BootstrapFilter(pomdp, 100))

hist = simulate(HistoryRecorder(), pomdp, planner, up, BootstrapFilter(pomdp, 100))



step = hist[1]

simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100))


inchrome(D3Tree(planner.tree))

mean([simulate(RolloutSimulator(), pomdp, RandomPolicy(pomdp), BootstrapFilter(pomdp, 100)) for _=1:10]) #0.0678
mean([simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100)) for _=1:10])

simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100))


plot()
for p in rand(b, 100)
    plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
end

histogram(b.surface_particle_belief)
histogram(b3.surface_particle_belief)
histogram(1:101,b5.surface_particle_belief)


plot(1:101, b5.surface_particle_belief)

using Distributions
using Random


