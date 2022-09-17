using SpillpointAnalysis
using POMDPs
using POMDPTools
using Plots
using JLD2
pgfplotsx()

pomdp = SpillpointInjectionPOMDP()

s = rand(initialstate(pomdp))

s, o ,r = gen(pomdp, s, (:drill, 0.2))
s, o ,r = gen(pomdp, s, (:inject, 0.12))

p = render(pomdp, s, return_one=true)

savefig(p, "examples/reservoir_example.tex")

using JLD2, SpillpointAnalysis


name1 = "POMCPOW (SIR)"
name2 = "Baseline (No Uncertainty)"

results1 = JLD2.load("results/trial_1/POMCPOW_SIR/state_1/results.jld2")
results2 = JLD2.load("results/trial_1/no_uncertainty/state_1/results.jld2")


trapped1 = [s.v_trapped / trap_capacity(s.m) for s in results1["states"]]
exited1 = [s.v_exited  for s in results1["states"]]

trapped2 = [s.v_trapped / trap_capacity(s.m) for s in results2["states"]]
exited2 = [s.v_exited for s in results2["states"]]


using Plots
using LaTeXStrings
pgfplotsx()

p1 = plot(trapped1, title=latexstring("Trapped CO\$_2\$"), label="$name1", color=1, palette=:bluesreds,legend=:topleft, ylabel="Trap Efficiency", size=(600,200))
plot!(trapped2, label="$name2", color=3,)


p2 = plot(exited1, title=latexstring("Exited CO\$_2\$"), label="$name1", color=1, palette=:bluesreds, size=(600,200), ylabel="Exited Volume", xlabel="Iteration", legend=:topleft)
plot!(exited2, label="$name2", color=3)

plot(p1, p2, layout=(2,1), size=(600,400))

savefig("examples/performance.tex")

