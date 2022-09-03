struct PerturbationResampler
	sampler
	perturb_fn
end

function ParticleFilters.resample(r::PerturbationResampler, ps, rng)
	ps = resample(r.sampler, ps, rng)
	unique_frac = length(unique([p.m.h for p in particles(ps)])) / length(particles(ps))
	
	if unique_frac < 0.25
		perturbed_ps = r.perturb_fn.(particles(ps))
		return ParticleCollection(perturbed_ps) 
	else
		return ps
	end 
end 

