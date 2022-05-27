using POMDPs, POMDPModels
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration

using StaticArrays

struct GridWorldState 
    x::Int64 # x position
    y::Int64 # y position
    done::Bool # are we in a terminal state?
end

# initial state constructor
GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false)


# the grid world mdp type
mutable struct GridWorld <: MDP{GridWorldState, Symbol} # Note that our MDP is parametarized by the state and the action
    size_x::Int64 # x size of the grid
    size_y::Int64 # y size of the grid
    reward_states::Vector{GridWorldState} # the states in which agent recieves reward
    reward_values::Vector{Float64} # reward values for those states
    tprob::Float64 # probability of transitioning to the desired state
    discount_factor::Float64 # disocunt factor
end


#we use key worded arguments so we can change any of the values we pass in 
function GridWorld(;sx::Int64=10, # size_x
    sy::Int64=10, # size_y
    rs::Vector{GridWorldState}=[GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)], # reward states
    rv::Vector{Float64}=rv = [-10.,-5,10,3], # reward values
    tp::Float64=0.7, # tprob
    discount_factor::Float64=0.9)

return GridWorld(sx, sy, rs, rv, tp, discount_factor)
end

# we can now create a GridWorld mdp instance like this:
mdp = GridWorld()
#mdp.reward_states # mdp contains all the defualt values from the constructor

function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::GridWorldState, mdp::GridWorld)
    v = SVector{3,Float64}(s.x, s.y, convert(Float64,s.done))
    return v
end

function POMDPs.convert_s(::Type{GridWorldState}, v::AbstractVector{Float64}, mdp::GridWorld)
    s = GridWorldState(round(Int64, v[1]), round(Int64, v[2]), convert(Bool, v[3]))
end


aa = convert_s(::Type{V} where V <: AbstractVector{Float64},::S,::P)

s=GridWorldState(4,3)
V = AbstractVector{Float64}
extracted_vector = POMDPs.convert_s(V,s,mdp)


s_reconstructed = convert_s(GridWorldState, extracted_vector, mdp)






VERTICES_PER_AXIS = 10 # Controls the resolutions along the grid axis
grid = RectangleGrid(range(1, stop=100, length=VERTICES_PER_AXIS), range(1, stop=100, length=VERTICES_PER_AXIS), [0.0, 1.0]) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

@requirements_info LocalApproximationValueIterationSolver(interp) GridWorld() # Check if the solver requirements are met