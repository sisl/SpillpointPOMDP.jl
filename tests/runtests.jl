using SpillpointPOMDP
using POMDPGifs
using POMCPOW
using POMDPSimulators
using Random
using POMDPs
using POMDPTools

pomdp = SpillpointInjectionPOMDP()

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=100, criterion=MaxUCB(20.0), tree_in_info=true)
planner = solve(solver, pomdp)

# mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, RandomPolicy(pomdp)) # Random policy gif
mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, planner, BootstrapFilter(pomdp, 100))


## Tests regarding the convert_s functions
s = rand(initialstate_distribution(pomdp))
v2 = convert_s(Vector{Float64}, s, pomdp)


s, _, _ = gen(pomdp, s, (:drill, 0.5))
v2 = convert_s(Vector{Float64}, s, pomdp)

s, _, _ = gen(pomdp, s, (:inject, 0.5))

# render(pomdp, s)

s.m.x
s.m.h
ρchan = s.m.ρ*ones(length(s.m.x))
inj = zeros(length(s.m.x))
inj[s.m.x .== s.x_inj] .= 1

thickness = [SpillpointPOMDP.observe_depth(s.polys, xpt)[2] for xpt in s.m.x]

v1 = hcat(s.m.x, s.m.h, ρchan, inj, thickness)

v2 = convert_s(Vector{Float64}, s, pomdp)

@assert v1 == v2