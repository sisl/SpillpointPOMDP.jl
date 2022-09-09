using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPTools
using POMCPOW
using ParticleFilters
using D3Trees
using BSON

## Playing around with the POMDP

# Initialize the pomdp
pomdp = SpillpointInjectionPOMDP(exited_reward=-1000)

# Setup and run the solver
solver = POMCPOWSolver(tree_queries=1000, criterion=MaxUCB(20.0), tree_in_info=true, estimate_value=0, k_observation=1, alpha_observation=0.3)
planner = solve(solver, pomdp)

s0 = rand(initialstate(pomdp))
s = deepcopy(s0)

# up_basic = BootstrapFilter(pomdp, 1000)
# up = BasicParticleFilter(pomdp, PerturbationResampler(LowVarianceResampler(1000), perturb_surface), 1000)
up = SpillpointAnalysis.SIRParticleFilter(
	model=pomdp, 
	N=200, 
	state2param=SpillpointAnalysis.state2params, 
	param2state=SpillpointAnalysis.params2state,
	N_samples_before_resample=100,
    clampfn=SpillpointAnalysis.clamp_distribution,
	prior=SpillpointAnalysis.param_distribution(initialstate(pomdp)),
	# use_all_prior_obs = false
	elite_frac=0.3
)

b0 = initialize_belief(up, initialstate(pomdp))
b = deepcopy(b0)

# Plot the belief
plot_belief(b0, s)

renders = []
belief_plots = [plot_belief(b, s0, title="timestep: 0")]
trees=[]
i=1



ret = 0
while !isterminal(pomdp, s)
	a, ai = action_info(planner, b)
	push!(trees, ai[:tree])
	println("action: ", a)
	sp, o, r = gen(pomdp, s, a)
	ret += r
	push!(renders, render(pomdp, sp, a, timestep=i))
	println("observation: ", o)
   	# if a[1] in [:drill, :observe]
	t = @elapsed b = update(up, b, a, o)
	println("belief update time: ", t)
	# else
		# b = update(up_basic, b, a, o)
	# end
	s = deepcopy(sp)
	push!(belief_plots, plot_belief(b, s0, title="timestep: $i"))
	i=i+1
	if i > 50
		break
	end
end

ret

anim = @animate for p in belief_plots
   plot(p)
end

gif(anim, "beliefs.gif", fps=2)

belief_plots[end-1]

anim = @animate for p in renders
   plot(p)
end

gif(anim, "renders.gif", fps=2)

inchrome(D3Tree(trees[end]))

trees[end].a_child_lookup

total_exited = 0
anim = @animate for p in rand(trees[end].root_belief.particles, 100)
   sp, o, r = gen(pomdp, p, (:inject, 0.07))
   println(o[2])
   global total_exited += o[2]
   # render(pomdp, sp)
   # title!("r=$r")
end

total_exited

plot_belief(trees[end].root_belief, s0)


solver = POMCPOWSolver(tree_queries=1000, criterion=MaxUCB(20.0), tree_in_info=true, estimate_value=0, k_observation=1, alpha_observation=0.1)
planner = solve(solver, pomdp)
a, ai = action_info(planner, trees[end].root_belief)

inchrome(D3Tree(ai[:tree]))

   
gif(anim, "belief_renders.gif", fps=2)

MaxQ

