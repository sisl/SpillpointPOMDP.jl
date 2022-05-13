@with_kw mutable struct SubsurfaceDistribution
	x = collect(0:0.02:1)
	x_inj = 0.25
	# ρ = Distributions.Uniform(0.5, 1.5)
	# lobe_height = Distributions.Uniform(0.05, 0.25)
	# center_height = Distributions.Uniform(0.05, 0.5)
	# linear_amplitude = Distributions.Uniform(0.05, 0.5)
	particle_set::Matrix{Float64} = reduce(hcat, 
				[rand(Random.MersenneTwister(123),Distributions.Uniform(0.5,1.5),100),
				rand(Random.MersenneTwister(212),Distributions.Uniform(0.05,0.25),100), 
				rand(Random.MersenneTwister(32),Distributions.Uniform(0.05,0.5),100),
				rand(Random.MersenneTwister(2323),Distributions.Uniform(0.05,0.5),100)]
				)
	
	#particle_set::Matrix{Float64} = reduce(hcat, [collect(0.5:0.01:1.5), collect(0.05:0.002:0.25), collect(0.05:0.0045:0.5), collect(0.05:0.0045:0.5)])
	surface_particle_belief::Vector{Float64} = repeat([1/100],100)

	obs_wells_hist::Vector{Float64} = []
	# polys = []
	

end

struct SubsurfaceUpdater <: POMDPs.Updater
    spec::SpillpointInjectionPOMDP
    num_MC
    SubsurfaceUpdater(spec::SpillpointInjectionPOMDP;  num_MC = 100) = new(spec, num_MC)
end

function POMDPs.update(up::SubsurfaceUpdater, b::SubsurfaceDistribution,
                    action::Tuple{Symbol, Float64}, obs) # check type of obs

	# obs = the taken obs
	# o = observations of the MC

	if action[1] == :observe

		#push!(b.obs_wells_hist, action[2])

		w_list = []

		for k in 1:up.num_MC

			s = rand(b, [k])
			# obs_wells
			#println("s.obs_wells ", s.obs_wells)
			# up.spec = pomdp problem
			_, o, _ = gen(up.spec, s, action)
			# observations = [observe_depth(s.polys, x_well) for x_well in s.obs_wells]

			mismatch = Float64(euclidean(obs, o))
			#println("mismatch ", mismatch)

			

			𝐰 = exp(-mismatch/0.005)

			push!(w_list,𝐰)
		end 

		# println("mismatch ", mismatch)

		𝒞 = Categorical(normalize(w_list, 1))

		#println("𝒞 ",typeof(𝒞))

		b.surface_particle_belief = 𝒞.p
		push!(b.obs_wells_hist, action[2])


		return b

	else 

		return b

end
end




function Base.rand(ssd::SubsurfaceDistribution)
	x = ssd.x
	 
	par_idx = Distributions.rand(Categorical(ssd.surface_particle_belief))
	p_set = ssd.particle_set[par_idx,:]
	
	# Sample the shape parameters
	# c1 = rand(ssd.lobe_height)
	# c2 = rand(ssd.center_height)
	# c3 = rand(ssd.linear_amplitude)
	ρ = p_set[1]
	c1 = p_set[2]
	c2 = p_set[3]
	c3 = p_set[4]

	# rand 100 particles having 
	
	# Determine the shape
	h = c1*sin.(5π*x) .+ c2*sin.(π*x) .+ c3 * x
	
	# Set the end points to be considered leaking
	h[1] = maximum(h)
	h[end] = h[1]
	
	# Sampling the porosity
	# ρ = rand(ssd.ρ)
	
	# Create the mesh and determine the injection spill region
	m = SpillpointMesh(x, h, ρ)
	sr = spill_region(m, ssd.x_inj)

	
	SpillpointInjectionState(;m, sr, ssd.x_inj, obs_wells = ssd.obs_wells_hist,p_set=p_set)
end

function Base.rand(ssd::SubsurfaceDistribution, par_idx::Vector{Int64})
	
	par_idx = par_idx[1]
	x = ssd.x
	 
	
	p_set = ssd.particle_set[par_idx,:]
	
	# Sample the shape parameters
	# c1 = rand(ssd.lobe_height)
	# c2 = rand(ssd.center_height)
	# c3 = rand(ssd.linear_amplitude)
	ρ = p_set[1]
	c1 = p_set[2]
	c2 = p_set[3]
	c3 = p_set[4]

	# rand 100 particles having 
	
	# Determine the shape
	h = c1*sin.(5π*x) .+ c2*sin.(π*x) .+ c3 * x
	
	# Set the end points to be considered leaking
	h[1] = maximum(h)
	h[end] = h[1]
	
	# Sampling the porosity
	# ρ = rand(ssd.ρ)
	
	# Create the mesh and determine the injection spill region
	m = SpillpointMesh(x, h, ρ)
	sr = spill_region(m, ssd.x_inj)
	SpillpointInjectionState(;m, sr, ssd.x_inj, obs_wells = ssd.obs_wells_hist,p_set=p_set)
	#SpillpointInjectionState(;m, sr, ssd.x_inj)
end

Base.rand(ssd::SubsurfaceDistribution, d::Int) = [rand(ssd) for _ in 1:d]

Base.rand(rng::AbstractRNG, ssd::SubsurfaceDistribution) = rand(ssd)



