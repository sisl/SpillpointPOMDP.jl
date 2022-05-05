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

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP()

# Plot the belief and the ground truth state
b = initialstate(pomdp)
s0 = rand(b)
solver = POMCPOWSolver(tree_queries=100, criterion=MaxUCB(20.0), tree_in_info=true)
planner = solve(solver, pomdp)
# action(planner, b)

action_hist = []
V = 0
t = 1 
while !isterminal(pomdp, s0)
    global V
    global t
    # problem: state does not get updated - keeps on filling up because no history of volumes in belief
    a = action(planner, b) 
    println(a)
    sp,o,r = gen(pomdp, s0, a)
    s0 = sp
    V += POMDPs.discount(pomdp)^(t-1)*r
    t += 1
    push!(action_hist, a)
    POMDPModelTools.render(pomdp, s0, "")  
    
end