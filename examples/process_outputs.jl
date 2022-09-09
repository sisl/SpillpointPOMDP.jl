using JLD2
using Polyhedra
using SpillpointAnalysis
using Statistics
using Printf

folder = "results/trial_1"
params = JLD2.load("$folder/params.jld2")

solvers = [:random, :no_uncertainty, :fixed_schedule, :POMCPOW_basic, :POMCPOW_SIR]
solver_names = ["Random", "Best Guess MDP", "Fixed Schedule", "POMCPOW (Basic)", "POMCPOW (SIR)"]
Nstates = 1

function mean_std_table_entry(vec, suffix = " & ")
	vec_mean = mean(vec)
	vec_std = std(vec)
	@sprintf("\$%.2f \\pm %.2f\$ %s", vec_mean, vec_std, suffix)
end

function mean_table_entry(vec, suffix = " & ")
	vec_mean = mean(vec)
	@sprintf("\$%.2f\$ %s", vec_mean, suffix)
end


table_string = ""
for (solver, solver_name) in zip(solvers, solver_names)
	solver_folder = "$folder/$solver"
	
	returns = []
	leaked = []
	trapped_efficiencies = []
	total_observations = []
	for i=1:Nstates
		results = JLD2.load("$solver_folder/state_$i/results.jld2")
		push!(returns, results["return"])
		push!(total_observations, sum([a[1] == :observe ? length(a[2]) : 0 for a in results["actions"]]))
		push!(leaked, results["states"][end].v_exited > 0)
		push!(trapped_efficiencies, results["states"][end].v_trapped / trap_capacity(results["states"][end].m))
	end 
	table_string = string(table_string, "$solver_name & ")
	table_string = string(table_string, mean_std_table_entry(returns))
	table_string = string(table_string, mean_std_table_entry(total_observations))
	table_string = string(table_string, mean_table_entry(leaked))
	table_string = string(table_string, mean_std_table_entry(trapped_efficiencies, "\\\\\n"))
end

println(table_string)

