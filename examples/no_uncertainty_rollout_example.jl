using SpillpointAnalysis
using Random
using ParticleFilters
using MCTS
include("utils.jl")

tree_queries = 100
exploration_coefficient = 1.
dir = "results/no_uncertainty"

pomdp = SpillpointInjectionPOMDP()

Random.seed!(0)
s0 = rand(initialstate(pomdp))

Random.seed!(floor(Int64, time()))

## Get a sample that matches the drilling location
up = SpillpointAnalysis.SIRParticleFilter(
	model=pomdp, 
	N=200, 
	state2param=SpillpointAnalysis.state2params, 
	param2state=SpillpointAnalysis.params2state,
	N_samples_before_resample=100,
	clampfn=SpillpointAnalysis.clamp_distribution,
	prior=SpillpointAnalysis.param_distribution(initialstate(pomdp)),
	elite_frac=0.3
)
b0 = initialize_belief(up, initialstate(pomdp))
a = (:drill, 0.5)
s, o, r = gen(pomdp, s0, a)
b1 = update(up, b0, a, o)
sguess = rand(b1)

up = BootstrapFilter(pomdp, 1)
b = ParticleCollection([sguess])

solver = MCTSSolver(n_iterations=tree_queries, depth=20, exploration_constant=exploration_coefficient, estimate_value=0)
planner = solve(solver, pomdp)

no_uncertainty_policy(b, i, observations, args...) = begin
	if length(observations)>0 && sum(observations[end][1:2]) > 0
		return (:stop, 0.0)
	else
		s_root = rand(b)
		i==1 && return a
		return action(planner, s_root)
	end
end

simulate_and_save(pomdp, no_uncertainty_policy, s0, b, up, dir, true)

