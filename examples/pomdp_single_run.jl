using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPTools
using POMCPOW
using ParticleFilters
using D3Trees
using BSON
using Random

## Playing around with the POMDP
Nstates=10
Random.seed!(0)
sample_pomdp = SpillpointInjectionPOMDP()
initial_states = [rand(initialstate(sample_pomdp)) for i=1:Nstates]

# Initialize the pomdp
exited_reward_amount = -1000 #rand(exited_reward_amount_options)
exited_reward_binary = -1000 #rand(exited_reward_binary_options)
obs_rewards = [-0.3, -0.7] #rand(obs_rewards_options)
height_noise_std = 0.1 #rand(height_noise_std_options)
sat_noise_std = 0.1 #height_noise_std
exploration_coefficient=20.#rand(exploration_coefficient_options)
alpha_observation=0.3#rand(alpha_observation_options)
k_observation=1.0#rand(k_observation_options)
tree_queries=1000#rand(tree_queries_options)

pomdp = SpillpointInjectionPOMDP(;exited_reward_amount, exited_reward_binary, obs_rewards, height_noise_std, sat_noise_std)

# pomdp = SpillpointInjectionPOMDP(exited_reward_binary=-1000)

# Setup and run the solver
optmisitic_val_estimate(pomdp, s, h, steps) = 0.5*pomdp.trapped_reward*(trap_capacity(s.m, s.sr, lb=s.v_trapped, ub=0.3, rel_tol=1e-2, abs_tol=1e-3) - s.v_trapped)
solver = POMCPOWSolver(;tree_queries, criterion=MaxUCB(exploration_coefficient), tree_in_info=true, estimate_value=optmisitic_val_estimate, k_observation, alpha_observation)
planner = solve(solver, pomdp)

s0 = deepcopy(initial_states[1])
s = deepcopy(s0)

up = SpillpointAnalysis.SIRParticleFilter(
	model=pomdp, 
	N=200, 
	state2param=SpillpointAnalysis.state2params, 
	param2state=SpillpointAnalysis.params2state,
	N_samples_before_resample=100,
    clampfn=SpillpointAnalysis.clamp_distribution,
	prior=SpillpointAnalysis.param_distribution(initialstate(pomdp)),
	elite_frac=0.3,
	bandwidth_scale=.5,
	max_cpu_time=60
)

b0 = initialize_belief(up, initialstate(pomdp))
b = deepcopy(b0)

# Plot the belief
plot_belief(b0, s)

renders = Any[render(pomdp, s, timestep=0, belief=b)]
beliefs = []
# belief_plots = [plot_belief(b, s0, title="timestep: 0")]
trees=[]
i=1
ret = 0
while !isterminal(pomdp, s)
	a, ai = action_info(planner, b)
	push!(trees, ai[:tree])
	println("action: ", a)
	sp, o, r = gen(pomdp, s, a)
	ret += r
	
	println("observation: ", o)
   	# if a[1] in [:drill, :observe]
	t = @elapsed b = update(up, b, a, o)
	println("belief update time: ", t)
	push!(renders, render(pomdp, sp, a, timestep=i, belief=b))
	# else
		# b = update(up_basic, b, a, o)
	# end
	s = deepcopy(sp)
	# push!(belief_plots, plot_belief(b, s0, title="timestep: $i"))
	i=i+1
	if i > 50
		break
	end
end

ret


anim = @animate for p in renders
   plot(p)
end

gif(anim, "renders.gif", fps=2)

inchrome(D3Tree(trees[end-1]))

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

plot_belief(trees[end-1].root_belief, s0)


solver = POMCPOWSolver(tree_queries=1000, criterion=MaxUCB(20.0), tree_in_info=true, estimate_value=0, k_observation=1, alpha_observation=0.1)
planner = solve(solver, pomdp)
a, ai = action_info(planner, trees[end].root_belief)

inchrome(D3Tree(trees[2]))

   
gif(anim, "belief_renders.gif", fps=2)

MaxQ

