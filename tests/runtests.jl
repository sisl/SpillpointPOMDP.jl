using SpillpointAnalysis
using POMDPGifs
using POMCPOW
using POMDPSimulators

pomdp = SpillpointInjectionPOMDP()

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=100, criterion=MaxUCB(20.0), tree_in_info=true)
planner = solve(solver, pomdp)

# mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, RandomPolicy(pomdp)) # Random policy gif
mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, planner, BootstrapFilter(pomdp, 100))

