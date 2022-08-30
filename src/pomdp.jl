@with_kw mutable struct SpillpointInjectionState
	m::SpillpointMesh
	sr
	x_inj
	polys = []
	v_trapped = 0
	v_exited = 0
	injection_rate = 0
	obs_wells = []
	stop = false
end

@with_kw struct SpillpointInjectionPOMDP <: POMDP{SpillpointInjectionState, Tuple{Symbol, Float64}, AbstractArray}
	Δt = .1
	injection_rates = [0.005, 0.01, 0.02]
	obs_locations = collect(0:0.2:1)
	sat_noise_std = 0.02
	exited_rate_noise_std = 0.01
	obs_reward = -.1
	exited_reward = -10000
	trapped_reward = 100
	s0_dist = SubsurfaceDistribution()
end

function POMDPs.actions(m::SpillpointInjectionPOMDP)
	injection_actions = [(:inject, val) for val in m.injection_rates]
	observation_actions = [(:observe, pos) for pos in m.obs_locations]
	[(:stop, 0.0), injection_actions..., observation_actions...]
end

function POMDPs.gen(pomdp::SpillpointInjectionPOMDP, s, a, rng=Random.GLOBAL_RNG)
	stop = s.stop
	obs_wells = deepcopy(s.obs_wells)
	injection_rate = s.injection_rate

	if a[1] == :stop
		injection_rate=0
		stop=true
	elseif a[1] == :inject
		injection_rate=a[2]
	elseif a[1] == :observe
		obs_wells = [obs_wells..., a[2]]
	else
		@error "unrecognized action: $(a)"
	end
	
	total_injected = s.v_trapped + s.v_exited + pomdp.Δt * injection_rate
	polys, v_trapped, v_exited = inject(s.m, s.sr, total_injected)
	
	sp = SpillpointInjectionState(s; polys, v_trapped, v_exited, obs_wells, stop, injection_rate)
	
	return (sp=sp, o=rand(observation(pomdp, s, a, sp)), r=reward(pomdp, s, a, sp))
end

function POMDPs.observation(pomdp::SpillpointInjectionPOMDP, s, a, sp)
	if isempty(sp.obs_wells)
		return Deterministic([])
	else
		observations = Float64[]
		obs_stds = Float64[]
		for x_well in sp.obs_wells
			if x_well in [0.0, 1.0]
				exited_rate = sp.v_exited - s.v_exited
				push!(observations, exited_rate)
				push!(obs_stds, pomdp.exited_rate_noise_std)
			else
				push!(observations, observe_depth(sp.polys, x_well))
				push!(obs_stds, pomdp.sat_noise_std)
			end
		end
		return MvNormal(observations, Diagonal(obs_stds.^2))
	end
end

function POMDPs.reward(pomdp::SpillpointInjectionPOMDP, s, a, sp)
	Δexited = sp.v_exited - s.v_exited
	Δtrapped = sp.v_trapped - s.v_trapped
	new_well = a[1] == :observe
	
	pomdp.exited_reward*Δexited + pomdp.trapped_reward*Δtrapped + pomdp.obs_reward*new_well
end

POMDPs.discount(::SpillpointInjectionPOMDP) = 0.99

POMDPs.isterminal(m::SpillpointInjectionPOMDP, s::SpillpointInjectionState) = s.stop

POMDPs.initialstate(m::SpillpointInjectionPOMDP) = m.s0_dist

function POMDPTools.render(m::SpillpointInjectionPOMDP, s::SpillpointInjectionState, a=nothing; timestep=nothing)
	poly, v_trapped, v_exited = inject(s.m, s.sr, s.v_trapped+s.v_exited)
	x, h = s.m.x, s.m.h
	p3 = plot(x, h, legend = :topleft, label="", yrange=(0,1.0))
	# for i in 1:length(x)
	# 	scatter!([x[i]], [h[i]], color=s.m.SR[i], label="")
	# end
	for p in poly
		if Polyhedra.volume(p) != 0
			plot!(p, color=:green, label="")
		end
	end
	maxh = maximum(s.m.h)
	title=isnothing(a) ? "" : "action: $a"
	if !isnothing(timestep)
		title = string(title, " timestep: $timestep")
	end
	plot!([s.x_inj, s.x_inj], [maxh + 0.2, maxh], arrow=true, linewidth=4, color=:black, label="", title=title)
	obs_label = "" #"Observation"
	for o in s.obs_wells
		plot!([o, o], [maxh + 0.2, maxh], arrow=true, linewidth=4, color=:blue, label=obs_label)
		obs_label = ""
	end
	
	p4 = bar(["trapped", "exited"], [v_trapped, v_exited], title="C02 volume", label="")
	p3
	plot(p3, p4, size=(900,300))
end

POMDPTools.render(m::SpillpointInjectionPOMDP, step; kwargs...) = render(m, step[:s], step[:a])

