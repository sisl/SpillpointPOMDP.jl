using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPTools
using POMCPOW
using ParticleFilters
using D3Trees
using JLD2
using Random
using MCTS
include("utils.jl")

USE_PLOT = true
Nstates = 10
max_steps = 50
Ntrials = 1

optmisitic_val_estimate(pomdp, s, args...) = 0.5*pomdp.trapped_reward*(trap_capacity(s.m, s.sr, lb=s.v_trapped, ub=0.3, rel_tol=1e-2, abs_tol=1e-3) - s.v_trapped)


Random.seed!(0)
sample_pomdp = SpillpointInjectionPOMDP()
initial_states = [rand(initialstate(sample_pomdp)) for i=1:Nstates]
solvers = [:no_uncertainty, :fixed_schedule, :POMCPOW_basic, :POMCPOW_SIR]

# trial = parse(Int, ARGS[1])
# println("running trial $trial")
# Random.seed!(trial)

exited_reward_amount_options = [-1000., -10000.]
exited_reward_binary_options = [-1000.]
obs_rewards_options = [[-.3, -.7], [-3., -7.]]
height_noise_std_options = [0.01, 0.1, 1.]
sat_noise_std_options = [0.01, 0.1, 1.]
exploration_coefficient_options = [2., 20.]
alpha_observation_options = [0.1, 0.3, 0.7]
k_observation_options=[1.,10.]
tree_queries_options=[1000, 5000]


## Run the experiments
try mkdir("results") catch end

for trial in 1:Ntrials
	println("starting trial $trial")
	trialdir = "results/trial_$trial"
	try mkdir(trialdir) catch end

	exited_reward_amount = -1000 
	exited_reward_binary = -1000 
	obs_rewards = [-0.3, -0.7] 
	height_noise_std = 0.01 
	sat_noise_std = 0.01 
	exploration_coefficient=20.
	alpha_observation=0.3
	k_observation=1.0
	tree_queries=5000
	
	exited_reward_amount = rand(exited_reward_amount_options)
	exited_reward_binary = rand(exited_reward_binary_options)
	obs_rewards = rand(obs_rewards_options)
	height_noise_std = rand(height_noise_std_options)
	sat_noise_std = height_noise_std
	exploration_coefficient = rand(exploration_coefficient_options)
	alpha_observation = rand(alpha_observation_options)
	k_observation = rand(k_observation_options)
	tree_queries = rand(tree_queries_options)

	params = Dict("exited_reward_amount"=>exited_reward_amount, 
					  "exited_reward_binary" => exited_reward_binary,
					  "obs_rewards"=>obs_rewards, 
					  "height_noise_std"=>height_noise_std, 
					  "sat_noise_std"=>sat_noise_std, 
					  "exploration_coefficient"=>exploration_coefficient,
					  "alpha_observation"=>alpha_observation,
					  "k_observation"=>k_observation,
					  "tree_queries"=>tree_queries)
	JLD2.save("$trialdir/params.jld2", params)

	# Initialize the pomdp
	pomdp = SpillpointInjectionPOMDP(;exited_reward_amount, exited_reward_binary, obs_rewards, height_noise_std, sat_noise_std)

	for solver_type in solvers
		println("kicking off solver: $solver_type")
		solver_dir = "$trialdir/$solver_type"
		try mkdir(solver_dir) catch end
		for (si, s0) in enumerate(initial_states)
			println("kicking off state: $si")
			dir = "$solver_dir/state_$si"
			
			if solver_type == :random
				random_policy(b, i, observations, s) = begin
					if length(observations)>0 && sum(observations[end][1:2]) > 0
						return (:stop, 0.0)
					else
						return rand(actions(pomdp, s))
					end
				end
				simulate_and_save(pomdp, random_policy, s0, nothing, nothing, dir, false, USE_PLOT)
				
			elseif solver_type == :no_uncertainty
				up = SpillpointAnalysis.SIRParticleFilter(
					model=pomdp, 
					N=400, 
					state2param=SpillpointAnalysis.state2params, 
					param2state=SpillpointAnalysis.params2state,
					N_samples_before_resample=100,
					clampfn=SpillpointAnalysis.clamp_distribution,
					prior=SpillpointAnalysis.param_distribution(initialstate(pomdp)),
					elite_frac=0.3,
				)
				b0 = initialize_belief(up, initialstate(pomdp))
				a = (:drill, 0.5)
				s, o, r = gen(pomdp, s0, a)
				b1 = update(up, b0, a, o)
				sguess = rand(b1)

				up = BootstrapFilter(pomdp, 1)
				b = ParticleCollection([sguess])

				solver = MCTSSolver(n_iterations=tree_queries, depth=20, exploration_constant=exploration_coefficient, estimate_value=optmisitic_val_estimate)
				planner = solve(solver, pomdp)

				no_uncertainty_policy(b, i, observations, args...) = begin
					if length(observations)>0 && sum(observations[end][1:2]) > 0
						return (:stop, 0.0)
					else
						s_root = rand(b)
						i==1 && return a
						return action(planner, s_root)
					end
				end
				simulate_and_save(pomdp, no_uncertainty_policy, s0, b, up, dir, true, USE_PLOT)
			elseif solver_type == :fixed_schedule
				up = SpillpointAnalysis.SIRParticleFilter(
					model=pomdp, 
					N=400, 
					state2param=SpillpointAnalysis.state2params, 
					param2state=SpillpointAnalysis.params2state,
					N_samples_before_resample=100,
				    clampfn=SpillpointAnalysis.clamp_distribution,
					prior=SpillpointAnalysis.param_distribution(initialstate(pomdp)),
					elite_frac=0.3,
					bandwidth_scale=.5,
					max_cpu_time=120
				)
				b0 = initialize_belief(up, initialstate(pomdp))
				a = (:drill, 0.5)

				fixed_schedule_policy(b, i, observations, args...) = begin
					if length(observations)>0 && sum(observations[end][1:2]) > 0
						return (:stop, 0.0)
					else
						i==1 && return a
						i % 3 == 0 && return (:observe, pomdp.obs_configurations[end])
						
						for injection_rate in reverse(pomdp.injection_rates)
							all_good = true
							for i=1:10
								s = rand(b)
								sp, o, r = gen(pomdp, s, (:inject, injection_rate))
								if r < 0 
									all_good = false
									println(" found leakage w/ injection rate: $injection_rate")
									break
								end
							end
							if all_good
								return (:inject, injection_rate)
							end
						end
						
						return (:stop, 0.0)
					end
				end

				simulate_and_save(pomdp, fixed_schedule_policy, s0, b0, up, dir, true, USE_PLOT)
			elseif solver_type == :POMCPOW_basic
				solver = POMCPOWSolver(;tree_queries, criterion=MaxUCB(exploration_coefficient), tree_in_info=false, estimate_value=optmisitic_val_estimate, k_observation, alpha_observation)
				planner = solve(solver, pomdp)
				up = BootstrapFilter(pomdp, 2000)
				b0 = initialize_belief(up, initialstate(pomdp))
				simulate_and_save(pomdp, (b, args...) -> action(planner, b), s0, b0, up, dir, true)
			elseif solver_type == :POMCPOW_SIR
				solver = POMCPOWSolver(;tree_queries, criterion=MaxUCB(exploration_coefficient), tree_in_info=false, estimate_value=optmisitic_val_estimate, k_observation, alpha_observation)
				planner = solve(solver, pomdp)
				up = SpillpointAnalysis.SIRParticleFilter(
					model=pomdp, 
					N=400, 
					state2param=SpillpointAnalysis.state2params, 
					param2state=SpillpointAnalysis.params2state,
					N_samples_before_resample=100,
				    clampfn=SpillpointAnalysis.clamp_distribution,
					prior=SpillpointAnalysis.param_distribution(initialstate(pomdp)),
					elite_frac=0.3,
					bandwidth_scale=.5,
					max_cpu_time=120
				)
				b0 = initialize_belief(up, initialstate(pomdp))
				simulate_and_save(pomdp, (b, args...) -> action(planner, b), s0, b0, up, dir, true, USE_PLOT)
			else
				@error "unrecognized solver type: $solver_type"
			end
		end
	end
end

