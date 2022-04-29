function graph_1d(N)
	G = [[] for _ in 1:N]
	for i in 1:N
		if i+1 <= N
			push!(G[i], i+1)
		end
		if i-1 >= 1
			push!(G[i], i-1)
		end
	end
	G
end

function edges(G)
	E = []
	for i in 1:length(G)
		for n in G[i]
			push!(E, sort!([i, n]))
		end
	end
	unique(E)
end

function upslope_neighbors(G, h)
	N = length(G)
	U = [[] for _ in 1:N]
	for i=1:N
		hs = [h[n] for n in G[i]]
		hmax_i = argmax(hs)
		hmax = hs[hmax_i]
		if hmax > h[i]
			push!(U[i], G[i][hmax_i])
		end
	end
	U
end


function set_spill_region(SR, sr_index, G, U, max_i)
	SR[max_i] = sr_index
	nodes_to_explore = [max_i]
	while !isempty(nodes_to_explore)
		n = pop!(nodes_to_explore)
		neighbors = G[n]
		for neighbor in neighbors
			if n in U[neighbor]
				if SR[neighbor] != sr_index
					SR[neighbor] = sr_index
					push!(nodes_to_explore, neighbor)
				end
			end
		end
	end
	SR
end

function spill_regions(G, h)
	U = upslope_neighbors(G, h)
	N = length(G)
	SR = zeros(Int, N)
	maxes = findall(U .== [[]])
	UB = zeros(length(maxes))
	for (i, max_i) in zip(1:length(maxes), maxes)
		set_spill_region(SR, i, G, U, max_i)
		UB[i] = h[max_i]
	end
	SR, UB
end

function spill_edges(G, SR)
	edges(G)[findall([SR[e[1]] != SR[e[2]] for e in edges(G)])]
end

function spill_region_graph(G, SR, SE)
	SRG = [[] for _ in 1:maximum(SR)]
	for e in SE
		for n in e
			push!(SRG[SR[n]], e)
		end
	end
	SRG
end

function upslope_spill_regions(SR, SRG, h)
	SRE, USR = [], zeros(Int, maximum(SR))
	for (i, E) in zip(1:length(SRG), SRG)
		if isempty(E)
			push!(SRE, [])
			
			continue
		end
		# Set the spill region edge
		ei = argmax([min(h[e[1]], h[e[2]]) for e in E])
		push!(SRE, E[ei])

		# Set the upstream spill region index
		edge_nodes = SR[E[ei]]
		n_adj = edge_nodes[edge_nodes .!= i][1]
		USR[i] = n_adj
	end
	SRE, USR
end

function spill_region_connectivity(G, SR, h)
	SE = spill_edges(G, SR)
	SRG = spill_region_graph(G, SR, SE)
	upslope_spill_regions(SR, SRG, h)
end

struct SpillpointMesh
	x		# Position of the grid points
	h		# Height at each point
	G		# Node connectivity
	SR		# Spill region index of each node
	UB		# Upper bound of the spill region
	SRE		# Spill region edges
	USR		# Upslope spill region index
	SRB		# Spill region bound (indices of SRs that leak)
	ρ		# Porosity
end

function SpillpointMesh(x, h, ρ)
	G = graph_1d(length(x))
	SR, UB = spill_regions(G, h)
	SRE, USR = spill_region_connectivity(G, SR, h)
	SRB = [SR[1], SR[end]]
	
	SpillpointMesh(x, h, G, SR, UB, SRE, USR, SRB, ρ)
end

function spill_region(m::SpillpointMesh, x_inj)
	nr = findfirst(m.x .> x_inj)
	nl = nr - 1 
	
	n = [nr, nl][argmax([m.h[nr], m.h[nl]])]
	m.SR[n]
end

function merge_spill_regions(m::SpillpointMesh, sr)
	# Identify the connected spill region
	se = m.USR[sr]
	edge_nodes = m.SR[se]
	n_adj = edge_nodes[edge_nodes .!= sr][1]

	# Update upper bound
	UB_new = copy(m.UB)
	UB_new[sr] = min(m.h[se]...)

	# Relabel and combine nodes
	SR_new = copy(m.SR)
	SR_new[SR_new .== n_adj] .= sr

	# reconstruct spill region graphs
	SRE_new, USR_new = spill_region_connectivity(m.G, SR_new, m.h)
	
	SpillpointMesh(m.x, m.h, m.G, SR_new, UB_new, SRE_new, USR_new, m.SRB, m.ρ)
end

