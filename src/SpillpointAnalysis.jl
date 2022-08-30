module SpillpointAnalysis

	using POMDPs
	using Polyhedra
	using Optim
	using LazySets
	using Parameters
	using Distributions
	using LinearAlgebra
	using POMDPTools
	using Plots
	using Random

	export SpillpointMesh
	include("mesh.jl")

	export inject
	include("spillpoint.jl")
	
	export SpillpointInjectionPOMDP, SpillpointInjectionState
	include("pomdp.jl")

	export SubsurfaceDistribution, perturb_surface
	include("beliefs.jl")

end

