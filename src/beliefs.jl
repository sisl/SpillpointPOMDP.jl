@with_kw struct SubsurfaceDistribution
	x = collect(0:0.02:1)
	ρ = Distributions.Uniform(0.5, 1.5)
	lobe_height = Distributions.Uniform(0.05, 0.25)
	center_height = Distributions.Uniform(0.05, 0.5)
	linear_amplitude = Distributions.Uniform(0.05, 0.5)
end

function param_distribution(d::SubsurfaceDistribution)
	product_distribution([d.lobe_height, d.center_height, d.linear_amplitude, d.ρ])
end

#TODO: Make this a function of the subsurface distribution since it can change
function clamp_distribution(x)
	x[1] = clamp(x[1],0.05, 0.25)
	x[2] = clamp(x[2],0.05, 0.5)
	x[3] = clamp(x[3],0.05, 0.5)
	x[4] = clamp(x[4],0.5, 1.5)
	x
end

function plot_belief(b, s0=nothing; title="belief", legend=:topleft)
   plt = plot(title=title, ylims=(0,1), legend=legend)
   for (i, p) in enumerate(particles(b))
       plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label=i==1 ? "Particles" : "")
   end
   if !isnothing(s0)
	   plot!(s0.m.x, s0.m.h, color=:red, label="Ground truth")
   end
   plt
end

function construct_surface(x, lobe_height, center_height, linear_amplitude, porosity)
	# Determine the shape
	h = lobe_height*sin.(5π*x) .+ center_height*sin.(π*x) .+ linear_amplitude * x
	
	# Set the end points to be considered leaking
	h[1] = maximum(h)
	h[end] = h[1]
	
	# Create the mesh
	SpillpointMesh(x, h, porosity, params=(lobe_height, center_height, linear_amplitude))
end

function Base.rand(ssd::SubsurfaceDistribution)
	# Sample the shape parameters to construct the mesh
	m = construct_surface(ssd.x, rand(ssd.lobe_height), rand(ssd.center_height), rand(ssd.linear_amplitude), rand(ssd.ρ))
	
	SpillpointInjectionState(;m)
end

Base.rand(ssd::SubsurfaceDistribution, d::Int) = [rand(ssd) for _ in 1:d]

Base.rand(rng::AbstractRNG, ssd::SubsurfaceDistribution) = rand(ssd)

function perturb_surface(s::SpillpointInjectionState)
	ρ_dist = Distributions.Uniform(-0.05, 0.05)
	lobe_height_dist = Distributions.Uniform(-0.025, 0.025)
	center_height_dist = Distributions.Uniform(-0.025, 0.025)
	linear_amplitude_dist = Distributions.Uniform(-0.025, 0.025)
	
	x = s.m.x
	ρ = s.m.ρ
	lobe_height, center_height, linear_amplitude = s.m.params
	
	# perturb the params
	ρ = clamp(ρ + rand(ρ_dist), 0.5, 1.5)
	lobe_height = clamp(lobe_height + rand(lobe_height_dist), 0.05, 0.25)
	center_height = clamp(center_height + rand(center_height_dist), 0.05, 0.5)
	linear_amplitude = clamp(linear_amplitude + rand(linear_amplitude_dist), 0.05, 0.5)
	
	# construct the new mesh and get spill region
	m = construct_surface(x, lobe_height, center_height, linear_amplitude, ρ)
	sr = spill_region(m, s.x_inj)
	
	# Recompute the filled regions
	total_injected = s.v_trapped + s.v_exited
	polys, v_trapped, v_exited = inject(s.m, s.sr, total_injected)
	
	# Copy everything else from the original state + the new mesh and C02
	SpillpointInjectionState(s; m, sr, polys, v_trapped, v_exited)
end





