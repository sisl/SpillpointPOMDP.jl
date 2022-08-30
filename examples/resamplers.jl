struct PerturbationResampler
	sampler
	perturb_fn
end

function ParticleFilters.resample(r::PerturbationResampler, ps, rng)
	ps = resample(r.sampler, ps, rng)
	perturbed_ps = r.perturb_fn.(particles(ps))
	ParticleCollection(perturbed_ps)  
end 

