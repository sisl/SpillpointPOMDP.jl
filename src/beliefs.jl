@with_kw struct SubsurfaceDistribution
	x = collect(0:0.01:1)
	x_inj = 0.25
	ρ = Distributions.Uniform(0.5, 1.5)
	lobe_height = Distributions.Uniform(0.05, 0.25)
	center_height = Distributions.Uniform(0.05, 0.5)
	linear_amplitude = Distributions.Uniform(0.05, 0.5)
end

function Base.rand(ssd::SubsurfaceDistribution)
	x = ssd.x
	
	# Sample the shape parameters
	c1 = rand(ssd.lobe_height)
	c2 = rand(ssd.center_height)
	c3 = rand(ssd.linear_amplitude)
	
	# Determine the shape
	h = c1*sin.(5π*x) .+ c2*sin.(π*x) .+ c3 * x
	
	# Set the end points to be considered leaking
	h[1] = maximum(h)
	h[end] = h[1]
	
	# Sampling the porosity
	ρ = rand(ssd.ρ)
	
	# Create the mesh and determine the injection spill region
	m = SpillpointMesh(x, h, ρ)
	sr = spill_region(m, ssd.x_inj)
	
	SpillpointInjectionState(;m, sr, ssd.x_inj)
end

Base.rand(ssd::SubsurfaceDistribution, d::Int) = [rand(ssd) for _ in 1:d]

Base.rand(rng::AbstractRNG, ssd::SubsurfaceDistribution) = rand(ssd)



