@with_kw mutable struct SpillpointInjectionState
	m::SpillpointMesh
	sr = nothing
	x_inj = nothing
	polys = []
	v_trapped = 0
	v_exited = 0
	stop = false
end

## Functions to convert back and forth from a parameter vector and state (used by SIR Particle Filter)
function params2state(params, sref)
	m = construct_surface(sref.m.x, params...)
	sr = isnothing(sref.x_inj) ? nothing : spill_region(m, sref.x_inj)
	
	# NOTE: We are explicitly NOT re-injecting and computing polys. Its not needed and is costly
	SpillpointInjectionState(sref; m, sr, v_trapped=sref.v_trapped, v_exited=sref.v_exited)
end

function state2params(s)
	[s.m.params..., s.m.ρ]
end

@with_kw struct SpillpointInjectionPOMDP <: POMDP{SpillpointInjectionState, Tuple{Symbol, Any}, AbstractArray}
	Δt = .1
	drill_locations = collect(0.1:0.1:0.9)
	injection_rates = [0.01, 0.07]
	# obs_configurations =[[0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], collect(0.25:0.25:0.75), collect(0.125:0.125:0.875)]
	obs_configurations =[collect(0.25:0.25:0.75), collect(0.125:0.125:0.875)]
	# obs_rewards = [-.1, -.1, -.1, -.1, -.5, -1.0]
	obs_rewards = [-.3, -.7]
	height_noise_std = 0.1
	sat_noise_std = 0.02
	exited_reward_amount = -10000
	exited_reward_binary = -10
	trapped_reward = 100
	s0_dist = SubsurfaceDistribution()
	γ = 0.9
end

function POMDPs.actions(m::SpillpointInjectionPOMDP, belief)
	actions(m, rand(belief))
end

function POMDPs.actions(m::SpillpointInjectionPOMDP, state::SpillpointInjectionState)
	if isnothing(state.x_inj)
		return [(:drill, val) for val in m.drill_locations]
	else
		injection_actions = [(:inject, val) for val in m.injection_rates]
		observation_actions = [(:observe, config) for config in m.obs_configurations]
		return [(:stop, 0.0), injection_actions..., observation_actions...]
	end
end

function POMDPs.gen(pomdp::SpillpointInjectionPOMDP, s, a, rng=Random.GLOBAL_RNG)
	stop = s.stop
	injection_rate = 0
	obs_wells = []
	x_inj = s.x_inj
	sr = s.sr

	if a[1] == :observe
	elseif a[1] == :drill
		x_inj = a[2]
		sr = spill_region(s.m, x_inj)
	elseif a[1] == :stop
		injection_rate=0
		stop=true
	elseif a[1] == :inject
		injection_rate=a[2]
	else
		@error "unrecognized action: $(a)"
	end
	
	total_injected = s.v_trapped + s.v_exited + pomdp.Δt * injection_rate
	polys, v_trapped, v_exited = inject(s.m, s.sr, total_injected)
	
	sp = SpillpointInjectionState(s; sr, x_inj, polys, v_trapped, v_exited, stop)
	
	return (sp=sp, o=rand(observation(pomdp, s, a, sp)), r=reward(pomdp, s, a, sp))
end

function POMDPs.observation(pomdp::SpillpointInjectionPOMDP, s, a, sp)
	# Check for leakage on either side
	if isempty(sp.polys)
		exited = [Bernoulli(0), Bernoulli(0)]
	else
		xleft = minimum([p[1] for p in points(sp.polys[1])])
		xright = maximum([p[1] for p in points(sp.polys[end])])
		if sp.v_exited == 0
			exited = [Bernoulli(0), Bernoulli(0)]
		elseif sp.v_exited > 0 && xleft ≈ s.m.x[2]
			exited = [Bernoulli(1), Bernoulli(0)]
		elseif sp.v_exited > 0 && xright ≈ s.m.x[end-1]
			exited = [Bernoulli(0), Bernoulli(1)]
		else
			@error string("exited: ", sp.v_exited, " xleft: ", xleft, " xright ", xright)
		end
	end
	
	# Construct distributions
	if a[1] == :drill
		drill_loc = findfirst(s.m.x .>= a[2])
		topsurface = s.m.h[drill_loc]
		return product_distribution([exited..., Normal(topsurface, 0.001)])
	elseif a[1] == :observe
		dists = []
		for x_well in a[2]
			height, thickness = observe_depth(sp.polys, x_well)
			if thickness == 0
				push!(dists, Bernoulli(0))
				push!(dists, Bernoulli(0))
			else
				push!(dists, Normal(height, pomdp.height_noise_std))
				push!(dists, Normal(thickness, pomdp.sat_noise_std))
			end
		end
		return product_distribution([exited..., dists...])
	else
		return product_distribution(exited)
	end

end

function POMDPs.reward(pomdp::SpillpointInjectionPOMDP, s, a, sp)
	Δexited = sp.v_exited - s.v_exited
	Δtrapped = sp.v_trapped - s.v_trapped
	if a[1] == :observe
		obs_reward = pomdp.obs_rewards[findfirst([a[2]] .== pomdp.obs_configurations)]
	else
		obs_reward = 0
	end
	exited_penalty = pomdp.exited_reward_binary * (Δexited > eps(Float32))
	
	pomdp.exited_reward_amount*Δexited + pomdp.trapped_reward*Δtrapped + obs_reward + exited_penalty
end

POMDPs.discount(pomdp::SpillpointInjectionPOMDP) = pomdp.γ

POMDPs.isterminal(m::SpillpointInjectionPOMDP, s::SpillpointInjectionState) = s.stop

POMDPs.initialstate(m::SpillpointInjectionPOMDP) = m.s0_dist

function POMDPTools.render(m::SpillpointInjectionPOMDP, s::SpillpointInjectionState, a=nothing; belief=nothing, timestep=nothing, return_one=false)
	poly, v_trapped, v_exited = inject(s.m, s.sr, s.v_trapped+s.v_exited)
	x, h = s.m.x, s.m.h
	p3 = plot([1], palette=:blues, color=1, label=L"{\rm CO}_2")
	# for i in 1:length(x)
	# 	scatter!([x[i]], [h[i]], color=s.m.SR[i], label="")
	# end
	for p in poly
		if Polyhedra.volume(p) != 0
			plot!(p, label="", palette=:blues, color=1, linecolor=1)
		end
	end
	plot!(x, h, legend = :topright, linewidth=2, label="Top Surface", color=!isnothing(belief) ? :red : :black, yrange=(0,1.0), ylabel="Height", xlabel="Position")
	
	maxh = maximum(s.m.h)
	title=isnothing(a) ? "Reservoir Top Surface" : "action: $a"
	if !isnothing(timestep)
		title = string(title, " timestep: $timestep")
	end
	if !isnothing(s.x_inj)
		plot!([s.x_inj, s.x_inj], [maxh + 0.2, maxh], arrow=true, linewidth=4, color=:black, label="Injector", title=title)
	end
	if !isnothing(a) && a[1] == :observe
		for (i, o) in enumerate(a[2])
			plot!([o, o], [maxh + 0.2, maxh], arrow=true, linewidth=4, color=:blue, label=i==1 ? "Observation Loc" : "")
		end
	end
	if !isnothing(belief)
		for p in belief.particles
	        plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
	    end
	end
	
	p4 = bar(["trapped", "exited"], [v_trapped, v_exited], title="C02 volume", label="")
	return_one ? p3 : plot(p3, p4, size=(900,300))
end

POMDPTools.render(m::SpillpointInjectionPOMDP, step; kwargs...) = render(m, step[:s], step[:a])

