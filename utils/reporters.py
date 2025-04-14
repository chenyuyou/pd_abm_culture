# <ATTACHMENT_FILE>
# <FILE_INDEX>File 2</FILE_INDEX>
# <FILE_NAME>reporters.py</FILE_NAME>
# <FILE_CONTENT>
import numpy as np # Import numpy

def get_cooperation_rate(model):
    """Calculate the fraction of cooperating agents in the model."""
    agent_count = model.schedule.get_agent_count()
    if agent_count == 0:
        return 0.0
    cooperator_count = sum([1 for agent in model.schedule.agents if agent.strategy == 1])
    return cooperator_count / agent_count

# --- Added Reporters for Culture ---
def get_average_culture(model):
    """Calculates the average cultural value C across all agents."""
    agent_count = model.schedule.get_agent_count()
    if agent_count == 0:
        return 0.0
    # Ensure agent.C exists and is numeric
    culture_values = [agent.C for agent in model.schedule.agents if hasattr(agent, 'C')]
    if not culture_values:
        return 0.0 # Or handle as an error/NaN if appropriate
    return np.mean(culture_values)

def get_std_culture(model):
    """Calculates the standard deviation of cultural value C."""
    agent_count = model.schedule.get_agent_count()
    if agent_count < 2: # Need at least 2 agents to calculate std dev
        return 0.0
    culture_values = [agent.C for agent in model.schedule.agents if hasattr(agent, 'C')]
    if len(culture_values) < 2:
        return 0.0
    return np.std(culture_values)


# new

# Example for reporters.py (needs refinement based on how you define type)
def get_segregation_index(model, threshold=0.5):
     # Assumes two types: A (C < threshold), B (C >= threshold)
     # Or better: use initial C if you store it, or define fixed types
     segregation_values = []
     for agent in model.schedule.agents:
         neighbors = model.grid.get_neighbors(agent.pos, moore=True, include_center=False)
         if not neighbors: continue
         
         # Define agent's type (EXAMPLE ONLY - ADAPT THIS)
         agent_type = 'A' if agent.C < threshold else 'B' 
         
         same_type_neighbors = 0
         for neighbor in neighbors:
             neighbor_type = 'A' if neighbor.C < threshold else 'B'
             if agent_type == neighbor_type:
                 same_type_neighbors += 1
         segregation_values.append(same_type_neighbors / len(neighbors))
     
     if not segregation_values: return 0.0
     return np.mean(segregation_values) 

# Example for reporters.py (needs refinement)
def get_cooperation_rate_by_type(model, cultural_type, threshold=0.5):
     # cultural_type: 'A' or 'B'
     # Define type logic (EXAMPLE ONLY - ADAPT THIS)
     group_agents = []
     for agent in model.schedule.agents:
          agent_c_type = 'A' if agent.C < threshold else 'B'
          if agent_c_type == cultural_type:
               group_agents.append(agent)

     if not group_agents: return 0.0
     cooperator_count = sum([1 for agent in group_agents if agent.strategy == 1])
     return cooperator_count / len(group_agents)

def get_cooperation_rate_A(model):
    return get_cooperation_rate_by_type(model, 'A')

def get_cooperation_rate_B(model):
    return get_cooperation_rate_by_type(model, 'B')


# --- End Added Reporters ---

# </FILE_CONTENT>
# </ATTACHMENT_FILE>
