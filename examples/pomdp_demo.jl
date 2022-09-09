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

Nstates = 10
Ntrials = 1000

Random.seed!(0)
pomdp = SpillpointInjectionPOMDP()
initial_states = [rand(initialstate(pomdp)) for i=1:Nstates]
updaters = [:basic, :SIR]

exited_reward_options = [-1000, -10000]
obs_rewards_options = [[-.1, -.5], [-1, -5]]
height_noise_std_options = [0.01, 0.1, 1]
sat_noise_std_options = [0.01, 0.1, 1]
exploration_coefficient_options = [2, 20]
alpha_observation_options = [0.1, 0.3, 0.7]
k_obsservation_options=[1,10]
tree_queries_options=[1000, 5000]

try mkdir("results") catch end


Threads.@threads for trial=1:Ntrials
	println("starting trial $trial")
	trialdir = "results/trial_$trial"
	try mkdir(trialdir) catch end

	exited_reward = rand(exited_reward_options)
	obs_rewards = rand(obs_rewards_options)
	height_noise_std = rand(height_noise_std_options)
	sat_noise_std = height_noise_std
	exploration_coefficient=rand(exploration_coefficient_options)
	alpha_observation=rand(alpha_observation_options)
	k_observation=rand(k_obsservation_options)
	tree_queries=rand(tree_queries_options)

	params = Dict("exited_reward"=>exited_reward, 
					  "obs_rewards"=>obs_rewards, 
					  "height_noise_std"=>height_noise_std, 
					  "sat_noise_std"=>sat_noise_std, 
					  "exploration_coefficient"=>exploration_coefficient,
					  "alpha_observation"=>alpha_observation,
					  "k_observation"=>k_observation,
					  "tree_queries"=>tree_queries)
   BSON.@save string(trialdir, "/params.bson") params	  

	try
	# Initialize the pomdp
	pomdp = SpillpointInjectionPOMDP(;exited_reward, obs_rewards, height_noise_std, sat_noise_std)

	for updater in updaters
		println("kicking off updater: $updater")
		dir = "$trialdir/$updater"
		try mkdir(dir) catch end
		for (si, s0) in enumerate(initial_states)
			s = deepcopy(s0)
			println("kicking off state: $si")
			# Setup and run the solver
			solver = POMCPOWSolver(;tree_queries, criterion=MaxUCB(exploration_coefficient), tree_in_info=false, estimate_value=0, k_observation, alpha_observation)
			planner = solve(solver, pomdp)

			if updater == :basic 
				up = BootstrapFilter(pomdp, 2000)
			elseif updater == :SIR
				up = SpillpointAnalysis.SIRParticleFilter(
					model=pomdp, 
					N=200, 
					state2param=SpillpointAnalysis.state2params, 
					param2state=SpillpointAnalysis.params2state,
					N_samples_before_resample=100,
				   clampfn=SpillpointAnalysis.clamp_distribution,
					prior=SpillpointAnalysis.param_distribution(initialstate(pomdp)),
					elite_frac=0.3
				)
			else
				@error "Unrecognized updater $updater"
			end

			b = initialize_belief(up, initialstate(pomdp))

			renders = []
			beliefs = [b]
			actions = []
			states = [s]
			# belief_plots = [plot_belief(b, s0, title="timestep: 0")]
			# trees=[]
			i=1
			ret = 0
			while !isterminal(pomdp, s)
				a, ai = action_info(planner, b)
				# push!(trees, ai[:tree])
				sp, o, r = gen(pomdp, s, a)
				ret += r
				push!(renders, render(pomdp, sp, a, timestep=i))
				push!(actions, a)
				push!(states, sp)
				println("action: $a, observation: $o")
				b = update(up, b, a, o)
				s = deepcopy(sp)
				# push!(belief_plots, plot_belief(b, s0, title="timestep: $i"))
				push!(beliefs, b)
				i=i+1
				if i > 50
					break
				end
			end

			# ret
			results = Dict("states"=>states, "actions"=>actions, "return"=>ret, "beliefs"=>beliefs)
			BSON.@save "$dir/results.bson" results

			# anim = @animate for p in belief_plots
			#    plot(p)
			# end
			# 
			# gif(anim, "$dir/beliefs.gif", fps=2)
			# 
			# anim = @animate for p in renders
			#    plot(p)
			# end
			# 
			# gif(anim, "$dir/renders.gif", fps=2)
		end
	end
	catch e
		BSON.@save "$dir/failure.bson" e
	end
end

