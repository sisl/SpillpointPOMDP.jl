using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPTools
using ParticleFilters
using D3Trees
using BSON

pomdp = SpillpointInjectionPOMDP(exited_reward=-1000)

basic_up = BootstrapFilter(pomdp, 1000)
up = SpillpointAnalysis.SIRParticleFilter(
	model=pomdp, 
	N=1000, 
	state2param=SpillpointAnalysis.state2params, 
	param2state=SpillpointAnalysis.params2state,
	N_samples_before_resample=100,
    clampfn=SpillpointAnalysis.clamp_distribution,
	prior=SpillpointAnalysis.param_distribution(initialstate(pomdp))
)

b0 = initialize_belief(up, initialstate(pomdp))
b = deepcopy(b0)

# Plot the belief
s0 = rand(initialstate(pomdp))
plot_belief(b0, s0)

a1 = (:drill, 0.5)
s1, o1, r1 = gen(pomdp, s0, a1)

b1_basic = update(basic_up, b0, a1, o1)
b1 = update(up, b0, a1, o1)

plot_belief(b1_basic, s0)
plot_belief(b1, s1)


