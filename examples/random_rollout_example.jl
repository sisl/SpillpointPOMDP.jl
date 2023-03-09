using SpillpointPOMDP
using Random

include("utils.jl")



pomdp = SpillpointInjectionPOMDP()

Random.seed!(0)
s0 = rand(initialstate(pomdp))

Random.seed!(floor(Int64, time()))

random_policy(b, i, observations, s) = begin
	if length(observations)>0 && sum(observations[end][1:2]) > 0
		return (:stop, 0.0)
	else
		return rand(actions(pomdp, s))
	end
end

simulate_and_save(pomdp, random_policy, s0, belief, nothing, "results/random_rollout_example", false)

