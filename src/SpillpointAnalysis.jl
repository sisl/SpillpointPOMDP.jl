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
	using Distances

	export SpillpointMesh
	include("mesh.jl")

	export inject
	include("spillpoint.jl")

	export SpillpointInjectionPOMDP
	include("pomdp.jl")

	export SubsurfaceDistribution
	export SubsurfaceUpdater
	include("beliefs.jl")



end

