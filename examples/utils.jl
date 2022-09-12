using JLD2
using POMDPs
using POMDPTools
using Plots

function simulate_and_save(pomdp, policy_fn, s0, b0, up, dir, update_belief=true, use_plot=true)
	try mkdir(dir) catch end # Make the directory 
	
	s = deepcopy(s0)
	b = deepcopy(b0)

	renders = Any[ render(pomdp, s, timestep=0, belief=update_belief ? b : nothing) ]
	beliefs = Any[b]
	actions = Any[]
	states = Any[s]
	observations = []
	
	i=1
	ret = 0
	prev_obs = []
	while !isterminal(pomdp, s)
		# Get the action
		tplan = @elapsed a = policy_fn(b, i, observations, s)
		
		# Gen the next step
		sp, o, r = gen(pomdp, s, a)
		
		# update belief
		if update_belief
			tbelief = @elapsed b = update(up, b, a, o)
		else
			tbelief = 0
		end
		
		# Store the other results
		ret += r
		use_plot && push!(renders, render(pomdp, sp, a, timestep=i, belief=update_belief ? b : nothing))
		push!(actions, a)
		push!(states, sp)
		push!(observations, o)
		update_belief && push!(beliefs, b)
		println("action: $a, observation: $o, planning time: ", tplan, " belief update_time: ", tbelief)
		
		# Prepare for next iteration
		s = deepcopy(sp)
		i=i+1
		i > max_steps && break
	end

	# Save the results
	results = Dict("states"=>states, "actions"=>actions, "return"=>ret, "beliefs"=>beliefs, "observations"=>observations)
	JLD2.save("$dir/results.jld2", results)

	# Plot the Gifs
	if use_plot
		anim = @animate for p in renders
		   plot(p)
		end
		gif(anim, "$dir/renders.gif", fps=2)
	end
end

