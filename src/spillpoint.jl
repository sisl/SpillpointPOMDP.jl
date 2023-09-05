issame(x1, y1) = x1[1] ≈ y1[1] && x1[2] ≈ y1[2]

function get_intersection(p1, p2, p3, p4)
	a = LineSegment(p1, p2)
	b = LineSegment(p3, p4)
	pt = intersection(a,b)
	if isempty(pt)
		@warn "Found no intersection with points: $p1, $p2, $p3, $p4. Seeing if any points are close..."
		if issame(p1, p3)
			return p1
		elseif issame(p1, p4)
			return p1
		elseif issame(p2, p3)
			return p2
		elseif issame(p2, p4)
			return p2
		end
	end
	if pt isa LineSegment
		return pt.p
	else
		return pt.element
	end
end

function get_polyhedron(m::SpillpointMesh, d, sr)
	# get all indices above the depth
	indices = findall((m.h .> d) .& (m.SR .== sr))
	l = max(1, indices[1] - 1)
	r = min(length(m.x), indices[end] + 1)

	hline = [[m.x[l], d], [m.x[r], d]]

	p1 = get_intersection([m.x[l], m.h[l]], [m.x[l+1], m.h[l+1]], hline...)
	p2 = get_intersection([m.x[r-1], m.h[r-1]], [m.x[r], m.h[r]], hline...)

	points = [p1, [[m.x[i], m.h[i]] for i in indices]..., p2]

	
	triangle1 = polyhedron(convexhull(points[1], points[2], [points[2][1], d]))
	polyhedra = [triangle1]
	
	for i=2:length(points)-1
		poly = polyhedron(convexhull(points[i], points[i+1], [points[i+1][1], d], [points[i][1], d]))
		push!(polyhedra, poly)
	end
	triangle2 = polyhedron(convexhull(points[end], points[end-1], [points[end-1][1], d]))
	push!(polyhedra, triangle2)
	polyhedra
end

function spill_region_volume(m::SpillpointMesh, d, sr)
	sum([Polyhedra.volume(p) for p in get_polyhedron(m, d, sr)])
end

function spill_region_max_poly(m::SpillpointMesh, sr)
	d = min(m.h[m.SRE[sr]]...)
	get_polyhedron(m, d, sr)
end

function spill_region_max_volume(m::SpillpointMesh, sr)
	sum([Polyhedra.volume(p) for p in spill_region_max_poly(m, sr)])
end

function get_depth(m::SpillpointMesh, sr, v)
	vol_diff(d) = (spill_region_volume(m, d, sr) - v)^2

	lo = min(m.h[m.SRE[sr]]...)
	if lo >= m.UB[sr]
		return lo
	end

	res = optimize(vol_diff, lo, m.UB[sr], rel_tol=1e-2, abs_tol=1e-3)
	Optim.minimizer(res)
end

function inject(m::SpillpointMesh, sr, v)
	v = v / m.ρ
	vtrapped = 0
	filled_srs = []
	polys = []
	
	while vtrapped < v
		vremain = v - vtrapped
		if spill_region_max_volume(m, sr) > vremain
			d = get_depth(m, sr, vremain)
			ps = get_polyhedron(m, d, sr)
			vtrapped = v
			push!(polys, ps...)
		else
			ps = spill_region_max_poly(m, sr)
			vtrapped += sum(Polyhedra.volume.(ps))
			push!(polys, ps...)
			
			push!(filled_srs, sr)
			new_sr = m.USR[sr]
			if new_sr in m.SRB
				return polys, m.ρ*vtrapped, m.ρ*(v - vtrapped)
			end
			if new_sr in filled_srs
				m = merge_spill_regions(m, sr)
				vtrapped = 0
				polys = []
				filled_srs = []
			else
				sr = new_sr
			end
		end
	end
	polys, m.ρ*vtrapped, 0
end

function get_bound(polygon, dim, fn)
	return fn([p[dim] for p in polygon.vrep.points.points])
end

function observe_depth(polygons, x_obs)
	for p in polygons
		xmin, xmax = get_bound(p, 1, extrema)
		if x_obs >= xmin && x_obs <= xmax
			ymin, ymax = get_bound(p, 2, extrema)
			pts = p.vrep.points.points
			vline = [[x_obs, ymax], [x_obs, ymin]]
			lpts = pts[[p[1] == xmin for p in pts]]
			rpts = pts[[p[1] == xmax for p in pts]]
				
			top_pts = [lpts[argmax([p[2] for p in lpts])], 
					   rpts[argmax([p[2] for p in rpts])]]
			ystart = get_intersection(top_pts..., vline...)[2]
			return ystart, ystart - ymin
		end
	end
	0, 0
end

function trap_capacity(m::SpillpointMesh)
	srs = unique(m.SR)[2:end-1]
	maximum([trap_capacity(m, sr) for sr in srs])
end

function trap_capacity(m::SpillpointMesh, sr; lb=0.0, ub=1.0, rel_tol=1e-2, abs_tol=1e-3)
	function max_trapped(v)
		_, v_trapped, v_exited = inject(m, sr, v)
		return -v_trapped + 1000*v_exited
	end
	res = optimize(max_trapped, lb, ub; rel_tol, abs_tol)
	Optim.minimizer(res)
end

