using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPTools
using POMCPOW
using ParticleFilters
using D3Trees
include("resamplers.jl")

function plot_belief(s0, b; title="belief")
   plt = plot(title=title)
   for p in b.particles
       plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
   end
   plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")
   plt
end

## Playing around with the POMDP

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP(exited_reward=-1000, sat_noise_std=0.01)

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=300, criterion=MaxUCB(2.0), tree_in_info=true, estimate_value=0)
planner = solve(solver, pomdp)

s0 = rand(initialstate(pomdp))
s = deepcopy(s0)
# up = BootstrapFilter(pomdp, 100)
up = BasicParticleFilter(pomdp, PerturbationResampler(LowVarianceResampler(100), perturb_surface), 100)
b0 = initialize_belief(up, initialstate(pomdp))
b = deepcopy(b0)


# Plot the belief
plot_belief(s, b0)

renders = []
belief_plots = []
trees=[]
i=0

ret = 0
while !isterminal(pomdp, s)
   a, ai = action_info(planner, b)
   push!(trees, ai[:tree])
   println("action: ", a)
   sp, o, r = gen(pomdp, s, a)
   ret += r
   push!(renders, render(pomdp, sp, a, timestep=i))
   println("observation: ", o)
   b, bi = update_info(up, b, a, o)
   s = deepcopy(sp)
   push!(belief_plots, plot_belief(s0, b, title="timestep: $i"))
   i=i+1
end

ret

anim = @animate for p in belief_plots
   plot(p)
end

gif(anim, "beliefs.gif", fps=2)

anim = @animate for p in renders
   plot(p)
end

gif(anim, "renders.gif", fps=2)

inchrome(D3Tree(trees[end]))

trees

belief_plots[end]



# Run two different solvers
mean([simulate(RolloutSimulator(), pomdp, RandomPolicy(pomdp), BootstrapFilter(pomdp, 100)) for _=1:10]) #0.0678
mean([simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100)) for _=1:10])

simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100))

