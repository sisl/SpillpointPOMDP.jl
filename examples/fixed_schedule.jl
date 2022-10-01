using SpillpointPOMDP
using Random
using ParticleFilters
include("utils.jl")

dir = "results/fixed_schedule"

pomdp = SpillpointInjectionPOMDP()

Random.seed!(0)
s0 = rand(initialstate(pomdp))

Random.seed!(floor(Int64, time()))

## Get a sample that matches the drilling location
up = SpillpointPOMDP.SIRParticleFilter(
	model=pomdp, 
	N=200, 
	state2param=SpillpointPOMDP.state2params,
	param2state=SpillpointPOMDP.params2state,
	N_samples_before_resample=100,
	clampfn=SpillpointPOMDP.clamp_distribution,
	prior=SpillpointPOMDP.param_distribution(initialstate(pomdp)),
	elite_frac=0.3
)
b0 = initialize_belief(up, initialstate(pomdp))
a = (:drill, 0.5)

fixed_schedule_policy(b, i, observations, args...) = begin
	if length(observations)>0 && sum(observations[end][1:2]) > 0
		return (:stop, 0.0)
	else
		i==1 && return a
		i % 3 == 0 && return (:observe, pomdp.obs_configurations[end])
		
		for injection_rate in reverse(pomdp.injection_rates)
			all_good = true
			for i=1:10
				s = rand(b)
				sp, o, r = gen(pomdp, s, (:inject, injection_rate))
				if r < 0 
					all_good = false
					println(" found leakage w/ injection rate: $injection_rate")
					break
				end
			end
			if all_good
				return (:inject, injection_rate)
			end
		end
		
		return (:stop, 0.0)
	end
end

simulate_and_save(pomdp, fixed_schedule_policy, s0, b0, up, dir, true)

