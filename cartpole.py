import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the CartPole environment
env = gym.make("CartPole-v1")

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("")
print("Number of actions:", env.action_space.n)

# Reset the environment to get initial observation
observation, info = env.reset(seed=42)
print(f"Initial observation: {observation}")
print("")

def render_episode(agent: Callable[[np.ndarray], int], agent_name: str = "Agent"):
    """
    Render an episode as an animation in the notebook.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    frames = []
    observation, info = env.reset()
    frames.append(env.render())

    total_reward = 0
    for step in range(500):
        action = agent(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    # Create animation
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"{agent_name} (Total Reward: {total_reward})")
    ax.axis('off')

    img = ax.imshow(frames[0])

    def animate(i):
        img.set_array(frames[i])
        return [img]

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)
    plt.close()

    return HTML(anim.to_jshtml())

def plot_episode_stats(agent: Callable, agent_name: str):
    """
    Plot the pole angle and cart position over an episode.
    """
    env = gym.make("CartPole-v1")
    observations, actions, rewards, total_reward = run_episode(env, agent)
    env.close()

    observations = np.array(observations)
    cart_positions = observations[:, 0]
    pole_angles = observations[:, 2]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.plot(cart_positions, linewidth=2)
    ax1.set_ylabel('Cart Position', fontsize=11)
    ax1.set_title(f'{agent_name} - Episode Performance (Total Reward: {total_reward})', fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(pole_angles, linewidth=2, color='orange')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Pole Angle (radians)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_episode(env: gym.Env, agent: Callable[[np.ndarray], int], max_steps: int = 500) -> tuple[list, list, list, float]:
    """
    Run a single episode using the given agent.
    
    Args:
        env: Gymnasium environment
        agent: Function that takes observation and returns action
        max_steps: Maximum steps per episode
        
    Returns:
        observations: List of observations
        actions: List of actions taken
        rewards: List of rewards received
        total_reward: Sum of all rewards
    """
    observations = []
    actions = []
    rewards = []
    
    # Reset environment for observation of initial state
    observation, info = env.reset()
    
    for step in range(max_steps):
        observations.append(observation)
        
        # Agent processes current game state and selects an action trajectory
        action = agent(observation)
        actions.append(action)
        
        # Action is executed, environment returns new game state, reward, and flags whether or not game end is necessary 
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        # Check whether episode must be ended for whatever reason
        if terminated or truncated:
            break
    
    total_reward = sum(rewards)
    return observations, actions, rewards, total_reward

def random_agent(observation: np.ndarray) -> int:
    return np.random.randint(0, 2)

#render_episode(random_agent, "Random Agent")

#plot_episode_stats(random_agent, "Random Agent")

def constant_agent(observation: np.ndarray) -> int:
    return 0  # Only go left

#render_episode(constant_agent, "Constant Agent")

#plot_episode_stats(constant_agent, "Constant Agent")

def heuristic_agent(observation: np.ndarray) -> int:
    
    pole_angle = observation[2]
    pole_velocity = observation[3]
    
    lean = pole_angle + 0.2*pole_velocity
    
    if lean > 0:
        return 1
    else:
        return 0

#render_episode(heuristic_agent, "Heuristic Agent")

#plot_episode_stats(heuristic_agent, "Heuristic Agent")

class SimpleNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # Create two linear layers (nn.Linear) and store them as class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # First layer + activation
        x = self.fc2(x)          # Output layer
        return x

# 4 inputs -> 16 hidden -> 2 outputs
net = SimpleNetwork(input_size=4, hidden_size=16, output_size=2)
x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
output = net(x)
loss = output.mean()
loss.backward()

# Create an optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Training loop pattern:
# 1. Zero out old gradients
optimizer.zero_grad()

# 2. Forward pass
output = net(x)
loss = output.mean()

# 3. Backward pass: compute ∇_θ loss
loss.backward()

# 4. Update parameters: θ_new = θ_old - α * ∇_θ loss
optimizer.step()

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)  # Input layer: obs_size -> hidden_size
        self.fc2 = nn.Linear(hidden_size, action_size)  # Output layer: hidden_size -> action_size
    
    def forward(self, x):
        # Forward pass
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        # Convert logits to probabilities using softmax
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n  # 2

policy_net = PolicyNetwork(obs_size=obs_size, hidden_size=32, action_size=action_size)

class NeuralAgent:
    def __init__(self, policy_network):
        self.policy_net = policy_network
    
    def __call__(self, observation: np.ndarray) -> int:
        
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_probs = self.policy_net(obs_tensor)
        
        action = torch.multinomial(action_probs, num_samples=1).item()
        
        return action

neural_agent = NeuralAgent(policy_net)

# Test the untrained agent
#obs, actions, rewards, total_reward = run_episode(env, neural_agent)
#print(f"Untrained agent reward: {total_reward}")

# Render it
#render_episode(neural_agent, "Untrained Neural Agent")

# Collect one episode
#observations, actions, rewards, total_reward = run_episode(env, neural_agent)

def compute_rewards_to_go(rewards: list[float], gamma: float = 0.99) -> list[float]:
    
    T = len(rewards)
    rewards_to_go = [0] * T
    
    rewards_to_go[T-1] = rewards[T-1]
    
    for t in range(T-2, -1, -1):
        rewards_to_go[t] = rewards[t] + gamma * rewards_to_go[t+1]
    
    return rewards_to_go

'''
# Test with simple example
test_rewards = [1, 1, 1, 10]
rtg = compute_rewards_to_go(test_rewards, gamma=0.9)
print(f"Rewards: {test_rewards}")
print(f"Rewards-to-go: {rtg}")
'''

def train_step(policy_net, optimizer, observations, actions, rewards, gamma=0.99):
    """
    Perform one REINFORCE training step on the policy network.
    
    Args:
        policy_net: the policy network
        optimizer: PyTorch optimizer
        observations: list of observation arrays
        actions: list of actions taken
        rewards: list of rewards received
        gamma: discount factor for reward-to-go
        
    Returns:
        loss: the computed loss value
    """
    # 1. Compute rewards-to-go
    rewards_to_go = compute_rewards_to_go(rewards, gamma)
    
    # 2-4. Convert to tensors
    obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
    action_tensor = torch.tensor(actions, dtype=torch.long)
    rtg_tensor = torch.tensor(rewards_to_go, dtype=torch.float32)
    
    # 5. Get action probabilities from network
    action_probs = policy_net(obs_tensor)
    
    # 6. Get probabilities of actions actually taken
    # gather selects the probability of the action that was actually taken
    taken_action_probs = action_probs.gather(1, action_tensor.unsqueeze(1)).squeeze()
    
    # 7. Compute log probabilities (add small epsilon for numerical stability)
    log_probs = torch.log(taken_action_probs + 1e-10)
    
    # 8. Compute REINFORCE loss: -E[log π(a|s) * R]
    loss = -(log_probs * rtg_tensor).mean()
    
    # 9. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 10. Return loss value
    return loss.item()

'''
# Test implementation, get loss after training step
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
loss = train_step(policy_net, optimizer, observations, actions, rewards)
'''

# Full training loop 

num_episodes = 200

policy_net = PolicyNetwork(obs_size=obs_size, hidden_size=32, action_size=action_size)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
neural_agent = NeuralAgent(policy_net)

episode_rewards = []

for episode in range(num_episodes):
    # Collect one episode
    observations, actions, rewards, total_reward = run_episode(env, neural_agent)
    
    # Train on this episode
    loss = train_step(policy_net, optimizer, observations, actions, rewards)
    
    # Track progress
    episode_rewards.append(total_reward)
    
    if episode % 20 == 0:
        print(f"Episode {episode}: Reward = {total_reward:.2f}, Loss = {loss:.4f}")

# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Training Progress')
plt.grid(True, alpha=0.3)
plt.show()

# Render the trained agent
render_episode(neural_agent, "Trained Neural Agent")

env.close()
