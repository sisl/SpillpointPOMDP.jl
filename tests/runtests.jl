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
solver = POMCPOWSolver(tree_queries=100, criterion=MaxUCB(1.0), tree_in_info=true)
planner = solve(solver, pomdp)

# mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, RandomPolicy(pomdp)) # Random policy gif
mygif = simulate(GifSimulator(filename="ccs3.gif"), pomdp, planner, BootstrapFilter(pomdp, 100))

inchrome(D3Tree(planner.tree))