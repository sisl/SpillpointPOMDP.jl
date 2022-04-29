module SpillpointAnalysis

	using POMDPs
	using Polyhedra
	using Optim
	using LazySets
	using Parameters
	using Distributions
	using LinearAlgebra
	using POMDPModelTools
	using Plots
	using Random

	export SpillpointMesh
	include("mesh.jl")

	export inject
	include("spillpoint.jl")

	export SubsurfaceDistribution
	include("beliefs.jl")

	export SpillpointInjectionPOMDP
	include("pomdp.jl")

end

