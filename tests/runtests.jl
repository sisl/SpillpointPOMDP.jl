using SpillpointPOMDP
using POMDPGifs
using POMCPOW
using POMDPSimulators
using Random
using POMDPs
using POMDPTools

pomdp = SpillpointInjectionPOMDP(obs_configurations = [[0.2, 0.5, 0.8], collect(0.1:0.1:0.9)],
                                 obs_rewards = [-.3, -.9])

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=100, criterion=MaxUCB(20.0), tree_in_info=true)
planner = solve(solver, pomdp)

# mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, RandomPolicy(pomdp)) # Random policy gif
mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, planner, BootstrapFilter(pomdp, 100))


## Tests regarding the convert_s functions
Random.seed!(0)
s = rand(initialstate_distribution(pomdp))
v2 = convert_s(Vector{Float64}, s, pomdp)

a = (:observe, [0.2, 0.5, 0.8])
avec = convert_a(Vector{Float64}, s, a, pomdp)

@assert all(avec[s.m.x .== 0.2, 3] .==  1)
@assert all(avec[s.m.x .== 0.5, 3] .==  1)
@assert all(avec[s.m.x .== 0.8, 3] .==  1)
@assert sum(avec) ==  3

a = (:drill, 0.5)
avec = convert_a(Vector{Float64}, s, a, pomdp)'

@assert avec[1, 26] ==  1
@assert sum(avec) ==  1

s, o, _ = gen(pomdp, s, (:drill, 0.5))
v2 = convert_s(Vector{Float64}, s, pomdp)

ovec = convert_o(Vector{Float64}, s, a, o, pomdp)
@assert ovec[26, 1] == 1
@assert ovec[26, 2] == o[3]
@assert sum(ovec) == 1 + o[3]

a = (:inject, 0.5)
avec = convert_a(Vector{Float64}, s, a, pomdp)

@assert avec[26, 2] ==  0.5
@assert sum(avec) ==  0.5

a = (:inject, 0.5)
s, o, _ = gen(pomdp, s, a)


ovec = convert_o(Vector{Float64}, s, a, o, pomdp)
@assert ovec[end, 3] == 1

a = (:observe, [0.2, 0.5, 0.8])
s, o, _ = gen(pomdp, s, a)
o
ovec = convert_o(Vector{Float64}, s, a, o, pomdp)

@assert all(ovec[s.m.x .== 0.2, 3] .== 1)
@assert all(ovec[s.m.x .== 0.5, 3] .== 1)
@assert all(ovec[s.m.x .== 0.8, 3] .== 1)

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