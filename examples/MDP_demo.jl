using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPModelTools
using POMCPOW
using POMDPSimulators
using ParticleFilters
using POMDPPolicies
using D3Trees
using Random

using LocalApproximationValueIteration
using LocalFunctionApproximation
using GridInterpolations


mdp_state = rand(SubsurfaceDistribution())

mdp = SpillpointInjectionPOMDP(s_ref_MDP=mdp_state, obs_locations=[])


grid = RectangleGrid(range(0, stop=0.2, length=10),
	                 [0,  mdp.injection_rates...],
                     [0, 1]);
                     
                     
interpolation = LocalGIFunctionApproximator(grid);

solver = LocalApproximationValueIterationSolver(interpolation,
											    max_iterations=100,
                                                verbose=true,
	        	  	 						    is_mdp_generative=true,
												n_generative_samples=1);


policy = solve(solver, mdp);

value(policy, mdp_state)


renders = []
i=0

s = mdp_state

while !isterminal(mdp, s)
   println("iteration: ", i)
   a, ai = action_info(policy, s)
   println("action: ", a)
   sp, _, r = gen(mdp, s, a)
   push!(renders, render(mdp, sp, timestep=i))
   println("reward: ", r)
   s = deepcopy(sp)
   i=i+1
end

anim = @animate for p in renders
    plot(p)
 end
 
 gif(anim, "images/renders.gif", fps=20)