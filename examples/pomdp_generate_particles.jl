using Revise
using SpillpointPOMDP
using Random

include("utils.jl")



pomdp = SpillpointInjectionPOMDP()

Random.seed!(0)
s0 = rand(initialstate(pomdp))

Random.seed!(floor(Int64, time()))

# random_policy(b, i, observations, s) = begin
# 	if length(observations)>0 && sum(observations[end][1:2]) > 0
# 		return (:stop, 0.0)
# 	else
# 		return rand(actions(pomdp, s))
# 	end
# end

# from Anthony "new":

random_policy(b, i, observations, s) = begin
    if length(observations)>0 && sum(observations[end][1:2]) > 0
        return (:stop, 0.0)
    else
        while true
            a = rand(actions(pomdp, s))
            a[1] != :stop && return a
        end
    end
end

up = SpillpointPOMDP.SIRParticleFilter(
    model=pomdp, 
    N=400, 
    state2param=SpillpointPOMDP.state2params,
    param2state=SpillpointPOMDP.params2state,
    N_samples_before_resample=100,
    clampfn=SpillpointPOMDP.clamp_distribution,
    prior=SpillpointPOMDP.param_distribution(initialstate(pomdp)),
    fraction_prior = 0.5,
    elite_frac=0.3,
    bandwidth_scale=.5,
    max_cpu_time=120
)
b0 = initialize_belief(up, initialstate(pomdp))

# results_all = Vector{Dict{String,Any}}()

simulate_and_save(pomdp, random_policy, s0, b0, nothing, "results/random_rollout_example", 1000, false)


# push!(results_all, results)

## testing
results = JLD2.load("/Users/markuszechner/Documents/code-dev/SpillpointPOMDP.jl/results/random_rollout_example/results.jld2")
sum(a["observations"][end][1:2])







# Questions for Anthony:
# - had to define a belief updater for sim - necessary for random? was not passed into the sim function before
# - definition of state: SpillpointInjectionState versus SpillpointInjectionPOMDP (noise on state?)
# - 1 roll-out (episode/trajectory/search,no/iteration,no)
# - random s0?

# - SLB meeting
# - optimization run numbers

