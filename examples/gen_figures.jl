using SpillpointAnalysis
using POMDPs
using POMDPTools
using Plots
pgfplotsx()

pomdp = SpillpointInjectionPOMDP()

s = rand(initialstate(pomdp))

s, o ,r = gen(pomdp, s, (:drill, 0.2))
s, o ,r = gen(pomdp, s, (:inject, 0.02))

p = render(pomdp, s, return_one=true)


savefig("reservoir_example.tex")


folder1 = 





