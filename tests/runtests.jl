using SpillpointAnalysis
using POMDPGifs
using POMCPOW
using POMDPSimulators

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

pomdp = SpillpointInjectionPOMDP()

# Setup and run the solver
#solver = POMCPOWSolver(tree_queries=100, criterion=MaxUCB(1.0), tree_in_info=true)
solver = POMCPOWSolver(tree_queries=100,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       k_action=3,
                       alpha_action=0.25,
                       k_observation=3,
                       #alpha_observation=0.15,
                       criterion=POMCPOW.MaxUCB(0.001),
                       final_criterion=POMCPOW.MaxQ(),
                       # final_criterion=POMCPOW.MaxTries(),
                       # estimate_value=0.0
                       )
planner = solve(solver, pomdp)

# mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, RandomPolicy(pomdp)) # Random policy gif
mygif = simulate(GifSimulator(filename="ccs5.gif"), pomdp, planner, BootstrapFilter(pomdp, 100))

# how to get final reward?
simulate(GifSimulator(filename="ccs4.gif"), pomdp, planner, BootstrapFilter(pomdp, 100))


inchrome(D3Tree(planner.tree))