using SpillpointAnalysis
using POMDPGifs

pomdp = SpillpointInjectionPOMDP()

mygif = simulate(GifSimulator(filename="ccs.gif"), pomdp, RandomPolicy(pomdp))

