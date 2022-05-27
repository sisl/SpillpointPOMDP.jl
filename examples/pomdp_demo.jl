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
using Random

function plot_belief(b; title)
   plt = plot(title=title)
   for p in b.particles
       plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
   end
   plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")
   plt
end

## Playing around with the POMDP

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP()

# Plot the belief and the ground truth state
Random.seed!(1222)
b = initialstate(pomdp)
s0 = rand(b)

# Plot the belief
#plot_belief(b, title="initial belief")

#Plotting beliefs

plot()
for p in rand(b, 100)
    plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
end
plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")

# # Here are the possible actions
# as = actions(pomdp)

# # We can call gen to start injecting
# sp1, o1, r1 = gen(pomdp, s0, (:inject, 0.1))

# # The observation is initially empty
# @assert o1 == []

# # We can then call gen to place an observation well at x=0.6
# sp2, o2, r2 = gen(pomdp, sp1, (:observe, 0.6))

# # We can see the observation well and the injected CO2 by rendering
# render(pomdp, sp2, "")

# # The resulting observations gives a noisy estimate of the amount of CO2 below
# o2

## Solving the POMDP

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=10000, 
                        k_action=2.0,
                        alpha_action=0.25,
                        k_observation=2.0,
                        alpha_observation=0.25,
                        criterion=POMCPOW.MaxUCB(0.1),
                        max_depth = 100, 
                        tree_in_info=true)

planner = solve(solver, pomdp)



s = s0
up = BootstrapFilter(pomdp, 100)
b0 = initialize_belief(up, initialstate(pomdp))
b = deepcopy(b0)

renders = []
belief_plots = []
trees=[]
i=0

while !isterminal(pomdp, s)
   println("iteration: ", i)
   a, ai = action_info(planner, b)
   push!(trees, ai[:tree])
   println("action: ", a)
   sp, o, r = gen(pomdp, s, a)
   push!(renders, render(pomdp, sp, timestep=i))
   println("observation: ", o)
   println("reward: ", r)
   b, bi = update_info(up, b, a, o)
   s = deepcopy(sp)
   push!(belief_plots, plot_belief(b, title="timestep: $i"))
   i=i+1
end







anim = @animate for p in belief_plots
   plot(p)
end

gif(anim, "images/beliefs.gif", fps=2)

anim = @animate for p in renders
   plot(p)
end

gif(anim, "images/renders.gif", fps=2)

inchrome(D3Tree(trees[1]))



# Run two different solvers
mean([simulate(RolloutSimulator(), pomdp, RandomPolicy(pomdp), BootstrapFilter(pomdp, 100)) for _=1:10]) #0.0678
mean([simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100)) for _=1:10])

simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100))

