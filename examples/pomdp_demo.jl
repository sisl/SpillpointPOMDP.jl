using SpillpointAnalysis
using Plots
using Distributions
using POMDPs
using POMDPTools
using POMCPOW
using ParticleFilters
using D3Trees
using BSON
include("resamplers.jl")

function plot_belief(s0, b; title="belief")
   plt = plot(title=title)
   for p in b.particles
       plot!(p.m.x, p.m.h, alpha=0.2, color=:gray, label="")
   end
   plot!(s0.m.x, s0.m.h, color=:red, label="ground truth")
   plt
end

## Playing around with the POMDP

planning_samples = [10, 100, 1000, 10000]
Ntrials = 10

for sz in planning_samples
   dir = "results/Nsamples_$sz"
   try mkdir(dir) catch end
   
   returns = []
   exited = []
   trapped = []
   for trial in 1:Ntrials

      # Initialize the pomdp
      pomdp = SpillpointInjectionPOMDP(exited_reward=-1000)

      # Setup and run the solver
      solver = POMCPOWSolver(tree_queries=sz, criterion=MaxUCB(2.0), tree_in_info=true, estimate_value=0)
      planner = solve(solver, pomdp)

      s0 = rand(initialstate(pomdp))
      s = deepcopy(s0)
      # up = BootstrapFilter(pomdp, 100)
      up = BasicParticleFilter(pomdp, PerturbationResampler(LowVarianceResampler(1000), perturb_surface), 1000)
      b0 = initialize_belief(up, initialstate(pomdp))
      b = deepcopy(b0)


      # Plot the belief
      # plot_belief(s, b0)

      renders = []
      belief_plots = []
      trees=[]
      i=0

      ret = 0
      try
      while !isterminal(pomdp, s)
         a, ai = action_info(planner, b)
         push!(trees, ai[:tree])
         println("action: ", a)
         sp, o, r = gen(pomdp, s, a)
         ret += r
         push!(renders, render(pomdp, sp, a, timestep=i))
         println("observation: ", o)
         b, bi = update_info(up, b, a, o)
         s = deepcopy(sp)
         push!(belief_plots, plot_belief(s0, b, title="timestep: $i"))
         i=i+1
         if i > 50
            break
         end
      end
      catch
         continue
      end

      data = Dict(:ret => ret, :s => s, :exited => s.v_exited, :trapped => s.v_trapped)
      BSON.@save "$(dir)/data_$trial.bson" data
      
      push!(returns, ret)
      push!(exited, s.v_exited)
      push!(trapped, s.v_trapped)

      anim = @animate for p in belief_plots
         plot(p)
      end

      gif(anim, "$(dir)/beliefs_$trial.gif", fps=2)

      anim = @animate for p in renders
         plot(p)
      end

      gif(anim, "$(dir)/renders_$trial.gif", fps=2)
   end
   data = Dict(:returns=>returns, :exited=>exited, :trapped=>trapped)
   BSON.@save "results/data_Nsamples_$sz.bson" data
end

data10 = BSON.load("results/data_Nsamples_10.bson")[:data]
data100 = BSON.load("results/data_Nsamples_100.bson")[:data]
data1000 = BSON.load("results/data_Nsamples_1000.bson")[:data]
data10000 = BSON.load("results/data_Nsamples_10000.bson")[:data]
datas = [data10, data100, data1000,]

function plot_metric(metric)
   mean_metrics = [mean(d[metric]) for d in datas]
   std_metrics = [std(d[metric]) for d in datas]
   plot([10, 100, 1000], mean_metrics, yerr=std_metrics, xscale=:log, line=false, marker=true, label="", ylabel="Return", xlabel="No. Tree Queries", title=metric)
end

p1 = plot_metric(:returns)
p2 = plot_metric(:trapped)
p3 = plot_metric(:exited)

plot(p1, p2, p3, layout = (1,3), size=(1800, 400))
savefig("No_Tree_Query_Results_r_1000.pdf")

# Run two different solvers
mean([simulate(RolloutSimulator(), pomdp, RandomPolicy(pomdp), BootstrapFilter(pomdp, 100)) for _=1:10]) #0.0678
mean([simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100)) for _=1:10])

simulate(RolloutSimulator(), pomdp, planner, BootstrapFilter(pomdp, 100))

