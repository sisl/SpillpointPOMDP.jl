@with_kw mutable struct SIRParticleFilter <: Updater
	model # The POMDP model
	N # Total particles to produce
	state2param # Function that maps a state to a vector of params
	param2state	# Funtion that maps a vector of params and a reference state to a state
	clampfn	# Clamps the parameters after sampling
	N_samples_before_resample = floor(Int, N/10) # Number of samples used between resamples
	fraction_prior = 0.0 # Fraction samples from prior
	min_bandwidth = 0.01 # Minimum bandwidth for the KDE (relevant when 1 good sample gets through)
	bandwidth_scale = 0.3 # Scale the bandwidth wrt to the silverman heuristic
	weight_clamp = 1 # Upper bound clamp on the importance weight
	elite_frac = 0.1 # For resampling, the elite fraction to use
	prior = nothing # Initial prior (used to compute importance weights)
	max_cpu_time = Inf
end

@with_kw mutable struct SIRParticleBelief{S}
	particle_collection::ParticleCollection{S} # store particle set here
	prior_observations::Vector{Any} = [] # store the prior observations here
end

ParticleFilters.particles(b::SIRParticleBelief) = particles(b.particle_collection)
ParticleFilters.weighted_particles(b::SIRParticleBelief) = weighted_particles(b.particle_collection)
ParticleFilters.support(b::SIRParticleBelief) = support(b.particle_collection)
# function ParticleFilters.rand(rng::AbstractRNG, b::SIRParticleBelief{S}) where S <: Any
# 	return SIRParticleBelief{S}(ParticleCollection{S}([rand(rng, b.particle_collection)]), b.prior_observations)
# end
function ParticleFilters.rand(rng::AbstractRNG, b::SIRParticleBelief{S}) where S <: Any
	return rand(rng, b.particle_collection)
end

function Base.rand(kde::KDEMulti)
	randrow = rand(1:length(kde.observations[1]))
	x = [rand(Normal(kde.observations[i][randrow], kde.KDEs[i].bandwidth)) for i=1:length(kde.dims)]
end

function POMDPs.initialize_belief(bu::SIRParticleFilter, D)
	particles = ParticleCollection(rand(D, bu.N))
	return SIRParticleBelief(particles, [])
end

function scale_bandwidth!(kde::KDEMulti, bandwidth_scale, min_bandwidth)
	for i=1:length(kde.KDEs)
		kde.KDEs[i].bandwidth *= bandwidth_scale
		kde.KDEs[i].bandwidth = max(kde.KDEs[i].bandwidth, min_bandwidth)
	end
end

function normalized_pdf(kde::KDEMulti, x)
	MultiKDE.pdf(kde, x) / length(kde.observations[1])
end

function POMDPs.update(up::SIRParticleFilter, b::Union{SIRParticleBelief{S},ParticleCollection{S}}, a, o) where S <: Any
	sref = particles(b)[1] # get a reference particle for converting back to the dynamic state

	# Store the observation (with associated state and action)
	if b isa SIRParticleBelief{S}
		prior_observations = deepcopy(b.prior_observations)
		use_all_prior_obs = true
	else # isa ParticleCollection{S}
		prior_observations = []
		use_all_prior_obs = false
	end
	push!(prior_observations, (sref, a, o))

	# Contruct the initial sampling distribution
	current_params = [up.state2param(p) for p in particles(b)] # should be d x N matrix where d is param dim and N is number of particles
	kde0 = KDEMulti([ContinuousDim() for _ in current_params[1]], nothing, current_params) # kernel density estimate of the prior
	kde = deepcopy(kde0)
	scale_bandwidth!(kde, up.bandwidth_scale, up.min_bandwidth)

	new_particle_states::Vector{S} = [] # Particles in their state form
	new_particle_params = [] # Particles in the parameter vector form
	poss = Float64[] # p(o | s)
	weights = Float64[] # Importance weights

	# plots = []
	tstart = time()
	# Loop until the number of particles is reached
	while true #length(new_particle_params) < 10*up.N
		if time() - tstart > up.max_cpu_time
			break
		end

		# Compute the number of samples from proposal dist
		N_prop = floor(Int,(1-up.fraction_prior)*up.N_samples_before_resample)

		# Take samples from proposal dist and compute weights
		for i=1:N_prop
			x = up.clampfn(rand(kde)) # Sample a new particle (and clamp it as necessary)

			pos = 1
			sp = nothing
			for (sref, a, o) in prior_observations
				s = up.param2state(x, sref) # Get the state from the parameter vector and refence state
				sp, _, _ = gen(up.model, s, a) # Propogate foward with the gen function

				pos *= Distributions.pdf(observation(up.model, s, a, sp), o) # Compute the observation weight p(o | s)
			end
			if use_all_prior_obs
				weight = clamp(Distributions.pdf(up.prior, x)  / normalized_pdf(kde, x), 0, up.weight_clamp) # Compute the importance weight (relative to the prior)
			else
				weight = clamp(normalized_pdf(kde0, x)  / normalized_pdf(kde, x), 0, up.weight_clamp)
			end

			# Store the particles in the arrays
			push!(new_particle_states, sp)
			push!(new_particle_params, x)
			push!(weights, weight)
			push!(poss, pos)
		end

		# Include some samples from prior
		for i=1:(up.N_samples_before_resample - N_prop)
			x = rand(up.prior)
			pos = 1
			sp = nothing
			for (sref, a, o) in prior_observations
				s = up.param2state(x, sref) # Get the state from the parameter vector and refence state
				sp, _, _ = gen(up.model, s, a) # Propogate foward with the gen function

				pos *= Distributions.pdf(observation(up.model, s, a, sp), o) # Compute the observation weight p(o | s)
			end


			# Store the particles in the arrays
			push!(new_particle_states, sp)
			push!(new_particle_params, x)
			push!(weights, 1.0)
			push!(poss, pos)
		end

		# p = plot(ylims=(0,1), title="count: $(length(new_particle_params))")
		# for particle in new_particle_states[end-up.N_samples_before_resample+1:end]
		# 	plot!(p, particle.m.h, color=:black, alpha=0.3, label="")
		# end

		# Construct a belief from the current set of particles
		weighted_ps = WeightedParticleBelief(new_particle_states, weights.*poss)
		obs = resample(LowVarianceResampler(up.N), weighted_ps, Random.GLOBAL_RNG)

		Nunique = length(unique([p.m.h for p in particles(obs)]))
		if Nunique >= 0.5*up.N || length(new_particle_states) > 100*up.N
			break
		end
		# println("Nunique: ", Nunique, " total: ", length(new_particle_states), " frac: ", Nunique / length(new_particle_states))

		Nelite = ceil(Int, up.N_samples_before_resample*up.elite_frac)
		# If we don't have enough unique samples, then we will just take the elite samples
		if Nunique < Nelite
			sorted_indices = sortperm(weights .* poss, rev=true)
			obs = ParticleCollection(new_particle_states[sorted_indices[1:Nelite]])
		end

		# for particle in particles(obs)
		# 	plot!(p, particle.m.h, color=:red, alpha=0.3, label="")
		# end
		# push!(plots, p)

		# Fit the next proposal using KDE
		obs = [up.state2param(p) for p in particles(obs)]
		if N_prop > 0
			kde = KDEMulti([ContinuousDim() for _ in obs[1]], nothing, obs)
			scale_bandwidth!(kde, up.bandwidth_scale, up.min_bandwidth)
		end
	end

	# Do the final resampling and return the particle set
	weighted_ps = WeightedParticleBelief(new_particle_states, weights.*poss)
	if sum(weights.*poss) == 0
		println("found all zero weights")
	end
	posterior_particles = resample(LowVarianceResampler(up.N), weighted_ps, Random.GLOBAL_RNG)
	if b isa SIRParticleBelief
		return SIRParticleBelief(posterior_particles, prior_observations)
	else
		return posterior_particles
	end
end