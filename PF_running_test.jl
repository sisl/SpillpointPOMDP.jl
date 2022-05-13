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
using POMDPGifs
using Serialization

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP()

# Plot the belief and the ground truth state
b = initialstate(pomdp)
#s0 = rand(b)
#serialize("s0_problem5", s0)
s0 = deserialize("s0_problem5")
solver = POMCPOWSolver(tree_queries=100, tree_in_info=true)
planner = solve(solver, pomdp)

up = SubsurfaceUpdater(pomdp)

# hist = simulate(GifSimulator(filename="ccs_new.gif"), pomdp, planner, up)

# for (s, b, a, r, sp, op) in hist
#     #println("s: $s, b: $(b.b), action: $a, obs: $op")
#     println("r: $r, action: $a")
# end
# println("Total reward: $(discounted_reward(hist))")

action_hist = []
belief_hist = []
reward_hist = []
render_hist = []
V = 0
t = 1

while !isterminal(pomdp, s0)
    global V
    global t
    println("step: ", t)
    push!(belief_hist, b)

    a = action(planner, b) 
    #println(a)
    sp, o, r = gen(pomdp, s0, a)

    #
    #render(pomdp, sp, "action: $a")
    # fix the seed
    # send Anthony the meshd
    # pic=(render(pomdp, deepcopy(sp), "action: $a"))
    # png(pic, string("image",t,".png"))
    display(render(pomdp, deepcopy(sp), ""))

    bp = update(up, deepcopy(b), a, o)
    

    s0 = sp
    b = bp

    V += POMDPs.discount(pomdp)^(t-1)*r
    t += 1
    push!(action_hist, a)
    push!(reward_hist, r)
        
end




# Running it with the simulate function directly:

#hist = simulate(HistoryRecorder(), pomdp, planner, up)


# Plotting beliefs
# plot()
# for p in rand(belief_hist[2], 100)
#     plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
# end
# plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")