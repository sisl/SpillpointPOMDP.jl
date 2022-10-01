using SpillpointPOMDP
using Plots
pgfplotsx()
using Distributions
using POMDPs
using POMDPTools
using ParticleFilters
using D3Trees
using BSON

pomdp = SpillpointInjectionPOMDP(height_noise_std = 0.1, sat_noise_std = 0.1)

basic_up = BootstrapFilter(pomdp, 200)
SIR_up = SpillpointPOMDP.SIRParticleFilter(
	model=pomdp, 
	N=200, 
	state2param=SpillpointPOMDP.state2params,
	param2state=SpillpointPOMDP.params2state,
	N_samples_before_resample=100,
    clampfn=SpillpointPOMDP.clamp_distribution,
	prior=SpillpointPOMDP.param_distribution(initialstate(pomdp)),
	elite_frac=0.3,
	bandwidth_scale=.5,
	max_cpu_time=60
)

b0_basic = initialize_belief(basic_up, initialstate(pomdp))
b0_SIR = initialize_belief(SIR_up, initialstate(pomdp))

# Plot the belief
s0 = rand(initialstate(pomdp))

plots = []

push!(plots, plot_belief(b0_basic, s0, title="Basic (0 Observations)"))
push!(plots, plot_belief(b0_SIR, s0, title="SIR (0 observations)", legend=false))

a1 = (:drill, 0.5)
s1, o1, r1 = gen(pomdp, s0, a1)

b1_basic = update(basic_up, b0_basic, a1, o1)
b1_SIR = update(SIR_up, b0_SIR, a1, o1)

push!(plots, plot_belief(b1_basic, s1, title="Basic (1 Observation)", legend=false))
push!(plots, plot_belief(b1_SIR, s1, title="SIR (1 Observation)", legend=false))


a2 = (:inject, 0.2)
s2, o2, r2 = gen(pomdp, s1, a2)

b2_basic = update(basic_up, b1_basic, a2, o2)
b2_SIR = update(SIR_up, b1_SIR, a2, o2)

a3 = (:observe, pomdp.obs_configurations[2])
s3, o3, r3 = gen(pomdp, s2, a3)

b3_basic = update(basic_up, b2_basic, a3, o3)
b3_SIR = update(SIR_up, b2_SIR, a3, o3)

push!(plots, plot_belief(b3_basic, s1, title="Basic (3 Observations)", legend=false))
push!(plots, plot_belief(b3_SIR, s1, title="SIR (3 Observations)", legend=false))
plot(plots..., layout=(3, 2), size=(900, 600))

savefig("SIR_basic_comparison.pdf")


