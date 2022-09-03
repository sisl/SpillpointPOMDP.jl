using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPTools
using POMCPOW
using ParticleFilters
using D3Trees
using BSON
include("resamplers.jl")

function plot_belief(s0, b; title="belief")
   plt = plot(title=title, ylims=(0,1))
   for p in b.particles
       plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
   end
   plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")
   plt
end

## Playing around with the POMDP

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP(exited_reward=-1000)

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=1000, criterion=MaxUCB(20.0), tree_in_info=true, estimate_value=0, k_observation=1, alpha_observation=0.3)
planner = solve(solver, pomdp)

s0 = rand(initialstate(pomdp))
s = deepcopy(s0)
plot_belief(s, b0)

# up = BootstrapFilter(pomdp, 1000)
up = BasicParticleFilter(pomdp, PerturbationResampler(LowVarianceResampler(1000), perturb_surface), 1000)
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
   if i > 50
      break
   end
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

trees[end].a_child_lookup

total_exited = 0
anim = @animate for p in rand(trees[end].root_belief.particles, 100)
   sp, o, r = gen(pomdp, p, (:inject, 0.07))
   println(o[2])
   global total_exited += o[2]
   # render(pomdp, sp)
   # title!("r=$r")
end

total_exited

plot_belief(s0, trees[end].root_belief)


solver = POMCPOWSolver(tree_queries=1000, criterion=MaxUCB(20.0), tree_in_info=true, estimate_value=0, k_observation=1, alpha_observation=0.1)
planner = solve(solver, pomdp)
a, ai = action_info(planner, trees[end].root_belief)

inchrome(D3Tree(ai[:tree]))

   
gif(anim, "belief_renders.gif", fps=2)

MaxQ

