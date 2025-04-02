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
# --- End Added Reporters ---

# </FILE_CONTENT>
# </ATTACHMENT_FILE>
