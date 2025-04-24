# reporters.py
import numpy as np
from collections import deque  # Needed for BFS in cluster finding

# === Standard Reporters (Keep As Is) ===


def get_cooperation_rate(model):
    """Calculate the fraction of cooperating agents (strategy=1) in the model."""
    agent_count = model.schedule.get_agent_count()
    if agent_count == 0:
        return 0.0
    # Ensure agents have 'strategy' attribute
    cooperator_count = sum([1 for agent in model.schedule.agents if hasattr(
        agent, 'strategy') and agent.strategy == 1])
    return cooperator_count / agent_count


def get_average_culture(model):
    """Calculates the average cultural value C across all agents."""
    agent_count = model.schedule.get_agent_count()
    if agent_count == 0:
        return 0.0
    # Ensure agent.C exists and is numeric
    culture_values = [
        agent.C for agent in model.schedule.agents if hasattr(agent, 'C')]
    if not culture_values:
        return 0.0  # Or handle as an error/NaN if appropriate
    return np.mean(culture_values)


def get_std_culture(model):
    """Calculates the standard deviation of cultural value C."""
    agent_count = model.schedule.get_agent_count()
    if agent_count < 2:  # Need at least 2 agents to calculate std dev
        return 0.0
    culture_values = [
        agent.C for agent in model.schedule.agents if hasattr(agent, 'C')]
    if len(culture_values) < 2:
        return 0.0
    return np.std(culture_values)

# === Segregation and Group-Specific Reporters (Keep/Refine) ===


def _get_agent_type(agent, threshold=0.5):
    """Helper function to determine cultural type based on threshold."""
    if not hasattr(agent, 'C'):
        return None  # Agent has no culture attribute
    return 'A' if agent.C < threshold else 'B'


def get_segregation_index(model, threshold=0.5):
    """
    Calculate the average fraction of neighbors of the same cultural type.
    This measures local similarity or clustering.
    Type A: C < threshold, Type B: C >= threshold.
    Returns the mean similarity across all agents with neighbors.
    """
    total_similarity = 0
    agents_with_neighbors = 0

    for agent in model.schedule.agents:
        agent_type = _get_agent_type(agent, threshold)
        if agent_type is None:
            continue  # Skip if agent has no type

        neighbors = model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False)
        if not neighbors:
            continue  # Skip agent if it has no neighbors

        same_type_neighbors = 0
        valid_neighbors = 0
        for neighbor in neighbors:
            neighbor_type = _get_agent_type(neighbor, threshold)
            if neighbor_type is not None:  # Only consider neighbors with a defined type
                valid_neighbors += 1
                if agent_type == neighbor_type:
                    same_type_neighbors += 1

        if valid_neighbors > 0:
            total_similarity += same_type_neighbors / valid_neighbors
            agents_with_neighbors += 1

    if agents_with_neighbors == 0:
        return 0.0  # Or np.nan, depending on desired handling
    return total_similarity / agents_with_neighbors


def get_cooperation_rate_by_type(model, cultural_type, threshold=0.5):
    """
    Calculate the cooperation rate for agents of a specific cultural type.
    cultural_type: 'A' or 'B'.
    """
    group_agents = []
    for agent in model.schedule.agents:
        agent_c_type = _get_agent_type(agent, threshold)
        if agent_c_type == cultural_type:
            group_agents.append(agent)

    if not group_agents:
        return 0.0  # No agents of this type found

    # Ensure agents have 'strategy' attribute
    cooperator_count = sum([1 for agent in group_agents if hasattr(
        agent, 'strategy') and agent.strategy == 1])
    return cooperator_count / len(group_agents)


def get_cooperation_rate_A(model, threshold=0.5):
    """Cooperation rate for Type A agents (C < threshold)."""
    return get_cooperation_rate_by_type(model, 'A', threshold)


def get_cooperation_rate_B(model, threshold=0.5):
    """Cooperation rate for Type B agents (C >= threshold)."""
    return get_cooperation_rate_by_type(model, 'B', threshold)


# === NEW Reporters for Physica A Analysis ===

# --- 1. Cluster Analysis ---
def get_cluster_size_distribution(model, threshold=0.5):
    """
    Calculates the size distribution of culturally homogeneous clusters.
    Uses Breadth-First Search (BFS) to find connected components of agents
    sharing the same cultural type ('A' or 'B').

    Args:
        model: The Mesa model instance.
        threshold (float): Threshold to differentiate Type A and Type B.

    Returns:
        dict: A dictionary where keys are cultural types ('A', 'B') and
              values are lists containing the sizes of all clusters found
              for that type. E.g., {'A': [1, 1, 5, 2, ...], 'B': [10, 3, 1, ...]}
              Returns {'A': [], 'B': []} if no agents exist.
    """
    cluster_sizes = {'A': [], 'B': []}
    visited = set()  # Store (x, y) tuples of visited agents

    # Create a mapping from position to agent for faster lookup during BFS
    pos_to_agent = {
        agent.pos: agent for agent in model.schedule.agents if agent.pos is not None}

    for agent in model.schedule.agents:
        if agent.pos is None or agent.pos in visited:
            continue

        agent_type = _get_agent_type(agent, threshold)
        if agent_type is None:
            visited.add(agent.pos)  # Mark as visited even if no type
            continue

        current_cluster_size = 0
        queue = deque([agent.pos])  # Queue for BFS, stores positions
        visited.add(agent.pos)

        while queue:
            current_pos = queue.popleft()
            current_cluster_size += 1

            # Find neighbors of the current position
            neighbor_coords = model.grid.get_neighborhood(
                current_pos, moore=True, include_center=False)

            for neighbor_pos in neighbor_coords:
                if neighbor_pos not in visited and neighbor_pos in pos_to_agent:
                    neighbor_agent = pos_to_agent[neighbor_pos]
                    neighbor_type = _get_agent_type(neighbor_agent, threshold)

                    # Check if neighbor is of the same type
                    if neighbor_type == agent_type:
                        visited.add(neighbor_pos)
                        queue.append(neighbor_pos)

        # Finished exploring a cluster
        cluster_sizes[agent_type].append(current_cluster_size)

    # Handle cases where one type might have no agents/clusters
    if not cluster_sizes['A']:
        cluster_sizes['A'] = []
    if not cluster_sizes['B']:
        cluster_sizes['B'] = []

    return cluster_sizes


# --- 2. Boundary Analysis ---

def get_boundary_fraction(model, threshold=0.5):
    """
    Calculates the fraction of agents located at the boundary between
    different cultural types ('A' vs 'B').
    An agent is on the boundary if at least one neighbor has a different type.
    """
    boundary_agent_count = 0
    total_agent_count = model.schedule.get_agent_count()

    if total_agent_count == 0:
        return 0.0

    for agent in model.schedule.agents:
        agent_type = _get_agent_type(agent, threshold)
        if agent_type is None:
            continue

        neighbors = model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False)
        if not neighbors:
            continue  # Skip agents without neighbors

        is_boundary = False
        for neighbor in neighbors:
            neighbor_type = _get_agent_type(neighbor, threshold)
            # If neighbor has a type AND it's different from the agent's type
            if neighbor_type is not None and neighbor_type != agent_type:
                is_boundary = True
                break  # Found one different neighbor, agent is on boundary

        if is_boundary:
            boundary_agent_count += 1

    return boundary_agent_count / total_agent_count


def get_boundary_coop_rate(model, threshold=0.5):
    """
    Calculates the cooperation rate specifically among agents located
    at the boundary between different cultural types.
    """
    boundary_agents = []
    for agent in model.schedule.agents:
        agent_type = _get_agent_type(agent, threshold)
        if agent_type is None:
            continue

        neighbors = model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False)
        if not neighbors:
            continue

        is_boundary = False
        for neighbor in neighbors:
            neighbor_type = _get_agent_type(neighbor, threshold)
            if neighbor_type is not None and neighbor_type != agent_type:
                is_boundary = True
                break

        if is_boundary:
            boundary_agents.append(agent)

    if not boundary_agents:
        return 0.0  # Or np.nan - No boundary agents found

    # Ensure agents have strategy attribute
    cooperator_count = sum([1 for agent in boundary_agents if hasattr(
        agent, 'strategy') and agent.strategy == 1])
    return cooperator_count / len(boundary_agents)


def get_bulk_coop_rate(model, threshold=0.5):
    """
    Calculates the cooperation rate specifically among agents located
    in the 'bulk' (i.e., NOT on the boundary).
    An agent is in the bulk if all its neighbors have the same cultural type.
    """
    bulk_agents = []
    for agent in model.schedule.agents:
        agent_type = _get_agent_type(agent, threshold)
        if agent_type is None:
            continue

        neighbors = model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False)

        # Agents with no neighbors are arguably not 'bulk' or 'boundary' - skip them?
        # Or consider them bulk? Let's skip them for now for a clearer definition.
        if not neighbors:
            continue

        is_bulk = True  # Assume bulk initially
        for neighbor in neighbors:
            neighbor_type = _get_agent_type(neighbor, threshold)
            # If neighbor has a type AND it's different, agent is NOT bulk
            if neighbor_type is not None and neighbor_type != agent_type:
                is_bulk = False
                break
            # If neighbor has NO type, is the agent still bulk? Ambiguous.
            # Let's assume bulk requires ALL neighbors to be same valid type.
            if neighbor_type is None:
                is_bulk = False
                break

        if is_bulk:
            bulk_agents.append(agent)

    if not bulk_agents:
        return 0.0  # Or np.nan - No bulk agents found

    # Ensure agents have strategy attribute
    cooperator_count = sum([1 for agent in bulk_agents if hasattr(
        agent, 'strategy') and agent.strategy == 1])
    return cooperator_count / len(bulk_agents)

# --- End NEW Reporters ---
