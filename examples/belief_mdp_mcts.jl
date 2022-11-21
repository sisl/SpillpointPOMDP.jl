using SpillpointPOMDP
using Plots
using Distributions
using POMDPs
using POMDPTools
using POMCPOW
using ParticleFilters
using D3Trees
using BSON
using Random
using MCTS


## Playing around with the POMDP
Nstates = 10
Random.seed!(0)
sample_pomdp = SpillpointInjectionPOMDP()
initial_states = [rand(initialstate(sample_pomdp)) for i=1:Nstates]

USE_POMCPOW = false

# Initialize the pomdp
exited_reward_amount = -1000 # rand(exited_reward_amount_options)
exited_reward_binary = -1000 # rand(exited_reward_binary_options)
obs_rewards = [-0.3, -0.7] # rand(obs_rewards_options)
height_noise_std = 0.1 # rand(height_noise_std_options)
sat_noise_std = 0.1 # height_noise_std
exploration_coefficient = 20.0 # rand(exploration_coefficient_options)
alpha_observation = 0.3 # rand(alpha_observation_options)
k_observation = 1.0 # rand(k_observation_options)
tree_queries = 10 # 1000 # rand(tree_queries_options)

pomdp = SpillpointInjectionPOMDP(;exited_reward_amount, exited_reward_binary, obs_rewards, height_noise_std, sat_noise_std)

up = SpillpointPOMDP.SIRParticleFilter(
	model=pomdp,
	N=USE_POMCPOW ?	200 : 100, # 200
	state2param=SpillpointPOMDP.state2params,
	param2state=SpillpointPOMDP.params2state,
	N_samples_before_resample=USE_POMCPOW ? 100 : 50, # 100
    clampfn=SpillpointPOMDP.clamp_distribution,
	fraction_prior=0.5,
	prior=SpillpointPOMDP.param_distribution(initialstate(pomdp)),
	elite_frac=0.3,
	bandwidth_scale=0.5,
	max_cpu_time=USE_POMCPOW ? 60 : 30
)

# Setup and run the solver
optmisitic_val_estimate(pomdp, s, h, steps) = 0.5*pomdp.trapped_reward*(trap_capacity(s.m, s.sr, lb=s.v_trapped, ub=0.3, rel_tol=1e-2, abs_tol=1e-3) - s.v_trapped)
optmisitic_val_estimate(bmdp::Union{BeliefMDP,GenerativeBeliefMDP}, b, depth) = mean(0.5*bmdp.pomdp.trapped_reward*(trap_capacity(s.m, s.sr, lb=s.v_trapped, ub=0.3, rel_tol=1e-2, abs_tol=1e-3) - s.v_trapped) for s in particles(b)) # TODO: Weighted mean?

s0 = deepcopy(initial_states[1])
s = deepcopy(s0)

if USE_POMCPOW
	solver = POMCPOWSolver(;tree_queries, criterion=MaxUCB(exploration_coefficient), tree_in_info=true, estimate_value=optmisitic_val_estimate, k_observation, alpha_observation)
	planner = solve(solver, pomdp)
else
	belief_reward(pomdp::POMDP, b, a, bp) = mean(reward(pomdp, s, a, sp) for (s,sp) in zip(particles(b), particles(bp))) # TODO: Are `s` and `sp` correct?
	bmdp = BeliefMDP(pomdp, up, belief_reward)
	solver = DPWSolver(; n_iterations=tree_queries, exploration_constant=exploration_coefficient, tree_in_info=true, estimate_value=optmisitic_val_estimate, show_progress=true)
	planner = solve(solver, bmdp)
end

b0 = initialize_belief(up, initialstate(pomdp))
b = deepcopy(b0)

# Plot the belief
plot_belief(b0, s) |> display

renders = Any[render(pomdp, s, timestep=0, belief=b)]
beliefs = []
trees = []
i = 1
ret = 0
# @time for _ in 1:10
@time while !isterminal(pomdp, s)
	global b, ret, i, trees, renders, s
	@info "Time: $i"

	a, ai = action_info(planner, b)
	push!(trees, ai[:tree])
	println("action: ", a)

	sp, o, r = gen(pomdp, s, a)
	ret += r

	println("observation: ", o)

	t = @elapsed b = update(up, b, a, o)
	println("belief update time: ", t)
	push!(renders, render(pomdp, sp, a, timestep=i, belief=b))
	renders[end] |> display
	
	s = deepcopy(sp)
	
	i = i+1
	i > 50 && break
end

@info ret

anim = @animate for p in renders
   plot(p)
end

gif(anim, "renders.gif", fps=2)

# renders[end] |> display

# inchrome(D3Tree(trees[end-1]))
# inchrome(D3Tree(trees[end]))
