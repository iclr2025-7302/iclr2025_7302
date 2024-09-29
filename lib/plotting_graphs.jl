include("./factor_graph.jl")
using Graphs
using GraphRecipes, Plots

# No-op unless implemented for the specialized nodes
function add_to_graph(node::Factor, g::SimpleDiGraph, id_in::Int, node_labels::Dict, edge_labels::Dict, pos::Dict)
    # No-op
    return id_out
end


# Creates the forward graph, but adds edge labels for both directions
function add_to_graph(node::GaussianLinearLayerFactor, g::SimpleDiGraph, id_in::Int, node_labels::Dict, edge_labels::Dict, pos::Dict)
    # Add nodes
    id_weights = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_weights] = "W"

    id_prod = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_prod] = "prod"

    id_sum = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_sum] = "sum"

    id_out = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_out] = "a"

    # Add edges:
    add_edge!(g, id_in, id_prod)
    # edge_labels[(id_in, id_prod)] = "$(short_str(node.last_m_from_v_in))"
    # edge_labels[(id_prod, id_in)] = "$(short_str(get_message_to_variable.(node.v_in)))"

    add_edge!(g, id_weights, id_prod)
    edge_labels[(id_weights, id_prod)] = "$(short_str(node.last_m_from_W))"
    edge_labels[(id_prod, id_weights)] = "$(short_str(get_message_to_variable.(node.W)))"

    add_edge!(g, id_prod, id_sum)
    edge_labels[(id_prod, id_sum)] = "$(short_str(node.m_edges))"
    edge_labels[(id_sum, id_prod)] = "$(short_str(node.m_edges_back))"

    add_edge!(g, id_sum, id_out)
    edge_labels[(id_sum, id_out)] = "$(short_str(get_message_to_variable.(node.v_out)))"
    edge_labels[(id_out, id_sum)] = "$(short_str(get_message.(node.v_out)))"

    # Positioning
    base_pos = pos[id_in]
    pos[id_weights] = base_pos + 0.9
    pos[id_prod] = base_pos + 1
    pos[id_sum] = base_pos + 2
    pos[id_out] = base_pos + 3

    return id_out
end

# Creates the forward graph, but adds edge labels for both directions
function add_to_graph(node::FirstGaussianLinearLayerFactor, g::SimpleDiGraph, id_in::Int, node_labels::Dict, edge_labels::Dict, pos::Dict)
    # Add nodes
    id_weights = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_weights] = "W"

    id_in = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_in] = "x"

    id_sum = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_sum] = "sum"

    id_out = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_out] = "a"

    # Add edges:
    add_edge!(g, id_in, id_sum)
    edge_labels[(id_in, id_sum)] = "$(round.(node.x, digits=2))"

    add_edge!(g, id_weights, id_sum)
    edge_labels[(id_weights, id_sum)] = "$(short_str(node.last_m_from_W))"
    edge_labels[(id_sum, id_weights)] = "$(short_str(get_message_to_variable.(node.W)))"

    add_edge!(g, id_sum, id_out)
    edge_labels[(id_sum, id_out)] = "$(short_str(get_message_to_variable.(node.v_out)))"
    edge_labels[(id_out, id_sum)] = "$(short_str(get_message.(node.v_out)))"

    # Positioning
    base_pos = 0
    pos[id_in] = base_pos
    pos[id_weights] = base_pos + 0.9
    pos[id_sum] = base_pos + 1
    pos[id_out] = base_pos + 2

    return id_out
end

function plot_branch(branch::Vector{Factor}, epoch::Int, it::Int; backward::Bool=false)
    g = SimpleDiGraph()
    node_labels, edge_labels, pos = Dict(), Dict(), Dict()
    id_out = -1
    for layer in branch
        # if (id_out == -1) && !(layer isa FirstGaussianLinearLayerFactor)
        #     id_out = 1 + nv(g)
        #     add_vertex!(g)
        #     node_labels[id_out] = "s"
        #     pos[id_out] = 1
        # end

        id_out = add_to_graph(layer, g, id_out, node_labels, edge_labels, pos)
    end

    # Int Positions are stretched to a LinkedList, whereas a float position means "round position, but put above the list". Sorry for the bad code
    x = [(isa(pos[i], Integer) ? pos[i] : round(pos[i])) for i in 1:nv(g)]
    y = [(isa(pos[i], Integer) ? 0 : 1) for i in 1:nv(g)]

    # Handle directions - the edge_labels should already contain labels for both directions
    if backward
        g = reverse(g)
    end

    node_names = [get(node_labels, i, " ") for i in 1:nv(g)]
    gp = graphplot(g; x, y, names=node_names, edgelabel=edge_labels, size=((1 + maximum(x)) * 200, 400), fontsize=10, nodeshape=:circle, nodesize=0.07)

    direction_label = (backward ? "Backward" : "Forward")
    p = plot(gp, title="Epoch $(epoch), Iteration $(it). $(direction_label)")
    display(p)
    savefig(p, "$(epoch)_$(it)_$(lowercase(direction_label)).png")
end

# Creates the forward graph, but adds edge labels for both directions
function add_to_graph(node::LeakyReLUFactor, g::SimpleDiGraph, id_in::Int, node_labels::Dict, edge_labels::Dict, pos::Dict)
    # Add nodes
    id_op = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_op] = "LR"

    id_out = 1 + nv(g)
    add_vertex!(g)
    node_labels[id_out] = "a"

    # Add edges:
    add_edge!(g, id_in, id_op)
    # edge_labels[(id_in, id_op)] = "$(short_str(get_message.(node.v_in)))"
    # edge_labels[(id_op, id_in)] = "$(short_str(get_message_to_variable.(node.v_in)))"

    add_edge!(g, id_op, id_out)
    edge_labels[(id_op, id_out)] = "$(short_str(get_message_to_variable.(node.v_out)))"
    edge_labels[(id_out, id_op)] = "$(short_str(get_message.(node.v_out)))"

    # Positioning
    base_pos = pos[id_in]
    pos[id_op] = base_pos + 1
    pos[id_out] = base_pos + 2

    return id_out
end