module SpillpointPOMDP

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
	using LaTeXStrings
	
	## For SIRParticleFilter
	using MultiKDE
	using ParticleFilters

	export SpillpointMesh
	include("mesh.jl")

	export inject, trap_capacity
	include("spillpoint.jl")
	
	export SpillpointInjectionPOMDP, SpillpointInjectionState
	include("pomdp.jl")

	export SubsurfaceDistribution, perturb_surface, plot_belief
	include("beliefs.jl")
	
	export SIRParticleFilter, SIRParticleBelief
	include("sir_particle_filter.jl")

end

