"""
PPO Training with Pest Management

This version adds pest management to your existing DSSAT-PPO setup.
It extends the action space to include pesticide applications.
"""

import gym
import gym_dssat_pdi
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# Pest simulator class (standalone, doesn't need DSSAT modifications)
class PestSimulator:
    """Simulates pest dynamics alongside DSSAT environment."""

    def __init__(self, config=None):
        self.config = config or {
            'base_pressure': 0.35,
            'weather_sensitivity': 0.6,
            'damage_rate': 0.025,
            'pesticide_efficacy': 0.7,
            'pesticide_cost': 15.0,
        }
        self.reset()

    def reset(self):
        """Reset pest state."""
        self.pest_pressure = self.config['base_pressure']
        self.cumulative_damage = 0.0
        self.days_since_pesticide = 999
        self.pesticide_applications = 0
        self.dap = 0

    def apply_pesticide(self):
        """Apply pesticide."""
        efficacy = self.config['pesticide_efficacy']
        self.pest_pressure *= (1 - efficacy)
        self.days_since_pesticide = 0
        self.pesticide_applications += 1

    def step(self, tmax=25, srad=20, apply_pesticide=False):
        """Update pest pressure for one day."""
        self.dap += 1

        if apply_pesticide:
            self.apply_pesticide()

        # Update days since pesticide
        self.days_since_pesticide += 1

        # Weather-based pest growth
        weather_sensitivity = self.config['weather_sensitivity']
        temp_factor = max(0, (tmax - 20) / 15)
        light_factor = max(0, (srad - 15) / 10)

        growth_rate = 0.05 * (1 + weather_sensitivity * (temp_factor + light_factor) / 2)

        # Pesticide decay
        decay = 0.0
        if self.days_since_pesticide < 14:
            decay = 0.02 * (14 - self.days_since_pesticide) / 14

        # Update pressure
        self.pest_pressure += growth_rate - decay
        self.pest_pressure = max(0.0, min(1.0, self.pest_pressure))

        # Accumulate damage
        damage_rate = self.config['damage_rate']
        daily_damage = damage_rate * self.pest_pressure
        self.cumulative_damage += daily_damage

        return {
            'pest_pressure': self.pest_pressure,
            'pest_damage': self.cumulative_damage,
            'days_since_pesticide': self.days_since_pesticide,
        }

    def get_cost(self):
        """Get total pesticide cost."""
        return self.pesticide_applications * self.config['pesticide_cost']


# helpers for action normalization
def normalize_action(action_space_limits, action):
    """Normalize the action from [low, high] to [-1, 1]"""
    low, high = action_space_limits
    return 2.0 * ((action - low) / (high - low)) - 1.0

def denormalize_action(action_space_limits, action):
    """Denormalize the action from [-1, 1] to [low, high]"""
    low, high = action_space_limits
    return low + (0.5 * (action + 1.0) * (high - low))


# Wrapper with pest management
class GymDssatWrapperWithPests(gym.Wrapper):
    """DSSAT wrapper with integrated pest management."""

    def __init__(self, env, enable_pests=True, pest_config=None):
        super(GymDssatWrapperWithPests, self).__init__(env)
        self.render_mode = None
        self.enable_pests = enable_pests

        # Initialize pest simulator
        if self.enable_pests:
            self.pest_sim = PestSimulator(pest_config)

        # Get original action space bounds
        self.action_low, self.action_high = self._get_action_space_bounds()

        # Extend action space for pesticide (binary: 0 or 1)
        num_actions = len(self.action_keys)
        if self.enable_pests:
            num_actions += 1  # Add pesticide action
            self.action_keys.append('pesticide')
            self.action_low = np.append(self.action_low, 0)
            self.action_high = np.append(self.action_high, 1)

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(num_actions,), dtype="float32"
        )

        # Extend observation space for pest variables
        obs_shape = env.observation_dict_to_array(env.observation).shape
        if self.enable_pests:
            # Add 3 pest variables: pressure, damage, days_since_pesticide
            obs_shape = (obs_shape[0] + 3,)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=obs_shape, dtype="float32"
        )

        self.last_info = {}
        self.last_obs = None

    def _get_action_space_bounds(self):
        action_keys = list(self.env.action_space.spaces.keys())

        lows = []
        highs = []
        for key in action_keys:
            box = self.env.action_space[key]
            lows.append(float(box.low))
            highs.append(float(box.high))

        self.action_keys = action_keys
        return np.array(lows), np.array(highs)

    def _format_action(self, action):
        """Format action for DSSAT (exclude pesticide)."""
        formatted = {}
        for i, key in enumerate(self.action_keys):
            if key == 'pesticide':
                continue  # Don't pass pesticide to DSSAT
            formatted[key] = action[i]
        return formatted

    def _format_observation(self, observation):
        """Format observation, adding pest variables."""
        obs_array = self.env.observation_dict_to_array(observation)

        if self.enable_pests:
            pest_vars = np.array([
                self.pest_sim.pest_pressure,
                self.pest_sim.cumulative_damage,
                min(self.pest_sim.days_since_pesticide, 30)  # Cap at 30
            ], dtype=np.float32)
            obs_array = np.concatenate([obs_array, pest_vars])

        return obs_array

    def reset(self, seed=None, options=None):
        """Reset environment and pest simulator."""
        if self.enable_pests:
            self.pest_sim.reset()
        return self._format_observation(self.env.reset())

    def step(self, action):
        """Step with pest management."""
        # Denormalize action
        denormalized_action = denormalize_action(
            (self.action_low, self.action_high), action
        )

        # Extract pesticide action
        apply_pesticide = False
        if self.enable_pests:
            pesticide_value = denormalized_action[-1]
            apply_pesticide = pesticide_value > 0.5  # Threshold
            denormalized_action = denormalized_action[:-1]  # Remove pesticide

        # Format DSSAT action
        formatted_action = self._format_action(denormalized_action)

        # Step DSSAT environment
        obs, reward, done, info = self.env.step(formatted_action)

        # Ensure info is a dictionary
        if info is None:
            info = {}

        # Get weather from observation
        obs_dict = obs if isinstance(obs, dict) else {}
        tmax = obs_dict.get('tmax', 25) if obs_dict else 25
        srad = obs_dict.get('srad', 20) if obs_dict else 20

        # Convert reward to scalar if needed
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = reward[0] if len(reward) > 0 else 0.0
        if reward is None:
            reward = 0.0
        reward = float(reward)

        # Update pest simulator
        if self.enable_pests:
            pest_state = self.pest_sim.step(tmax, srad, apply_pesticide)

            # Adjust reward for pest costs and damage
            pest_cost = self.pest_sim.get_cost()
            pest_damage_value = self.pest_sim.cumulative_damage * 10  # Scale damage
            reward = reward - pest_cost - pest_damage_value

            # Add pest info
            info['pest_pressure'] = pest_state['pest_pressure']
            info['pest_damage'] = pest_state['pest_damage']
            info['pesticide_applications'] = self.pest_sim.pesticide_applications

        # Handle done state
        if done:
            obs, reward, info = self.last_obs, 0.0, self.last_info
        else:
            self.last_obs = obs
            self.last_info = info

        formatted_observation = self._format_observation(obs)
        formatted_observation = np.array(formatted_observation, dtype=np.float32).flatten()

        return formatted_observation, float(reward), done, info

    def close(self):
        return self.env.close()

    def seed(self, seed):
        self.env.set_seed(seed)

    def __del__(self):
        self.close()


# Baseline agents
class NullAgent:
    """Agent always choosing to do nothing."""
    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        num_actions = self.env.action_space.shape[0]
        actions = np.zeros(num_actions, dtype=np.float32)
        return actions, obs


class ExpertAgent:
    """Expert agent with fixed schedule."""
    fertilization_dic = {40: 27, 45: 35, 80: 54}
    irrigation_dic = {20: 25, 50: 30, 80: 25}

    def __init__(self, env):
        self.env = env

    def _policy(self, obs):
        # Assume dap is first element
        dap = int(obs[1]) if len(obs) > 1 else 0

        actions = []
        for key in self.env.action_keys:
            if key == 'anfer':
                actions.append(self.fertilization_dic.get(dap, 0))
            elif key == 'amir':
                actions.append(self.irrigation_dic.get(dap, 0))
            elif key == 'pesticide':
                # Apply pesticide if pest pressure high (from observation)
                pest_pressure_idx = -3  # Third from end
                if len(obs) > abs(pest_pressure_idx):
                    pest_pressure = obs[pest_pressure_idx]
                    actions.append(1.0 if pest_pressure > 0.6 else 0.0)
                else:
                    actions.append(0.0)
            else:
                actions.append(0)

        return actions

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = self._policy(obs)
        action = normalize_action((self.env.action_low, self.env.action_high), action)
        return np.array(action, dtype=np.float32), obs


# Evaluation and plotting
def evaluate(agent, n_episodes=10, enable_pests=True):
    eval_args = {
        'mode': 'all',
        'seed': 1025,
        'random_weather': True,
    }
    env = Monitor(GymDssatWrapperWithPests(
        gym.make('GymDssatPdi-v0', **eval_args),
        enable_pests=enable_pests
    ))

    returns, _ = evaluate_policy(
        agent, env, n_eval_episodes=n_episodes, return_episode_rewards=True
    )

    env.close()
    return returns


def evaluate_with_pest_tracking(agent, agent_name, n_episodes=3, enable_pests=True):
    """Evaluate agent and track pest dynamics over episodes."""
    eval_args = {
        'mode': 'all',
        'seed': 1025,
        'random_weather': True,
    }

    env = GymDssatWrapperWithPests(
        gym.make('GymDssatPdi-v0', **eval_args),
        enable_pests=enable_pests
    )

    episode_histories = []

    for ep in range(n_episodes):
        history = {
            'dap': [],
            'pest_pressure': [],
            'pest_damage': [],
            'days_since_pesticide': [],
            'pesticide_applications': [],
            'rewards': [],
            # DSSAT crop state variables
            'grnwt': [],      # Grain weight
            'xlai': [],       # Leaf area index
            'canwaa': [],     # Canopy weight
            'vstage': [],     # Vegetative stage
        }

        obs = env.reset()
        done = False
        step = 0

        # Get base DSSAT environment - unwrap all wrappers
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
            # Check if this is the DSSAT environment
            if hasattr(base_env, '_state'):
                break

        # Debug: Print available attributes on first episode
        if ep == 0:
            print(f"  Debug - base_env type: {type(base_env)}")
            print(f"  Debug - has _state: {hasattr(base_env, '_state')}")
            if hasattr(base_env, '_state') and base_env._state:
                print(f"  Debug - _state keys sample: {list(base_env._state.keys())[:5] if isinstance(base_env._state, dict) else 'not a dict'}")

        while not done and step < 200:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Track pest metrics
            if enable_pests:
                history['dap'].append(env.pest_sim.dap)
                history['pest_pressure'].append(env.pest_sim.pest_pressure)
                history['pest_damage'].append(env.pest_sim.cumulative_damage)
                history['days_since_pesticide'].append(min(env.pest_sim.days_since_pesticide, 30))
                history['rewards'].append(reward)

                # Check if pesticide was applied this step
                if step > 0 and env.pest_sim.days_since_pesticide == 1:
                    history['pesticide_applications'].append(env.pest_sim.dap)

            # Track DSSAT crop state variables from env._state
            if hasattr(base_env, '_state') and base_env._state is not None:
                state = base_env._state
                # Track DAP from state if not already tracked from pest_sim
                if not enable_pests and 'dap' in state:
                    history['dap'].append(int(state.get('dap', step)))
                history['grnwt'].append(float(state.get('grnwt', 0)))
                history['xlai'].append(float(state.get('xlai', 0)))
                history['canwaa'].append(float(state.get('canwaa', 0)))
                history['vstage'].append(float(state.get('vstage', 0)))
            elif hasattr(base_env, 'observation') and base_env.observation is not None:
                # Fallback: try using observation dict
                obs_dict = base_env.observation
                if not enable_pests and 'dap' in obs_dict:
                    history['dap'].append(int(obs_dict.get('dap', step)))
                history['grnwt'].append(float(obs_dict.get('grnwt', 0)))
                history['xlai'].append(float(obs_dict.get('xlai', 0)))
                history['canwaa'].append(float(obs_dict.get('canwaa', 0)))
                history['vstage'].append(float(obs_dict.get('vstage', 0)))

            step += 1

        # Debug: Check if crop data was collected
        if ep == 0:
            print(f"  Debug - grnwt samples: {len(history['grnwt'])}, sample values: {history['grnwt'][:3] if history['grnwt'] else 'empty'}")

        episode_histories.append(history)

    env.close()
    return episode_histories


def plot_pest_impact(histories_dict, output_file='pest_impact.png'):
    """
    Create comprehensive pest impact visualization.

    Args:
        histories_dict: Dict mapping agent names to their episode histories
        output_file: Output filename for the plot
    """
    print(f"\nGenerating pest impact analysis graph...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'null': '#d62728', 'ppo': '#1f77b4', 'expert': '#2ca02c'}

    # Left: Pest Pressure with Applications
    for agent_name, histories in histories_dict.items():
        if histories:
            max_len = max(len(h['dap']) for h in histories)
            pressure_avg = []

            for i in range(max_len):
                values = [h['pest_pressure'][i] for h in histories if i < len(h['pest_pressure'])]
                if values:
                    pressure_avg.append(np.mean(values))

            dap = list(range(len(pressure_avg)))
            ax1.plot(dap, pressure_avg, label=agent_name, linewidth=3,
                    color=colors.get(agent_name, '#999'))

            # Mark applications
            for h in histories:
                for app_day in h['pesticide_applications']:
                    if app_day < len(pressure_avg):
                        ax1.scatter(app_day, pressure_avg[app_day], marker='v', s=120,
                                   color=colors.get(agent_name, '#999'), alpha=0.8,
                                   edgecolors='black', linewidth=1.5, zorder=5)

    ax1.axhline(0.6, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
    ax1.set_xlabel('Days After Planting', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Pest Pressure (0-1)', fontsize=13, fontweight='bold')
    ax1.set_title('Pest Pressure Over Time\n▼ = Pesticide Application', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)

    # Right: Cumulative Damage
    for agent_name, histories in histories_dict.items():
        if histories:
            max_len = max(len(h['dap']) for h in histories)
            damage_avg = []

            for i in range(max_len):
                values = [h['pest_damage'][i] for h in histories if i < len(h['pest_damage'])]
                if values:
                    damage_avg.append(np.mean(values))

            dap = list(range(len(damage_avg)))
            ax2.plot(dap, damage_avg, label=agent_name, linewidth=3,
                    color=colors.get(agent_name, '#999'))

            # Annotate final values
            if damage_avg:
                final_val = damage_avg[-1]
                ax2.text(len(damage_avg)+2, final_val, f'{final_val:.1f} kg/ha',
                        fontsize=11, va='center', color=colors.get(agent_name, '#999'),
                        fontweight='bold')

    ax2.set_xlabel('Days After Planting', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Damage (kg/ha)', fontsize=13, fontweight='bold')
    ax2.set_title('Pest Damage Accumulation', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=11)

    fig.suptitle('Pest Impact Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_pest_management_strategies(histories_dict, output_file='pest_management.png'):
    """
    Create pest management strategy comparison visualization.

    Args:
        histories_dict: Dict mapping agent names to their episode histories
        output_file: Output filename for the plot
    """
    print(f"\nGenerating pest management strategies graph...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'null': '#d62728', 'ppo': '#1f77b4', 'expert': '#2ca02c'}

    # Left panel: Application timing distribution
    for agent_name, histories in histories_dict.items():
        if histories:
            all_applications = []
            for h in histories:
                all_applications.extend(h['pesticide_applications'])

            if all_applications:
                ax1.hist(all_applications, bins=20, alpha=0.5, label=agent_name.upper(),
                        color=colors.get(agent_name, '#999'), edgecolor='black')

    ax1.set_xlabel('Days After Planting', fontsize=12)
    ax1.set_ylabel('Number of Applications', fontsize=12)
    ax1.set_title('Pesticide Application Timing', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right panel: Performance comparison (applications vs damage)
    applications_count = {}
    final_damages = {}

    for agent_name, histories in histories_dict.items():
        if histories:
            counts = [len(h['pesticide_applications']) for h in histories]
            applications_count[agent_name] = np.mean(counts)
            damages = [h['pest_damage'][-1] for h in histories]
            final_damages[agent_name] = np.mean(damages)

    agents = list(applications_count.keys())
    x_pos = np.arange(len(agents))
    width = 0.35

    ax2_twin = ax2.twinx()

    bars1 = ax2.bar(x_pos - width/2, [applications_count[a] for a in agents], width,
                    label='Pesticide Apps', color='steelblue', alpha=0.7)
    bars2 = ax2_twin.bar(x_pos + width/2, [final_damages[a] for a in agents], width,
                         label='Final Damage', color='coral', alpha=0.7)

    ax2.set_xlabel('Agent', fontsize=12)
    ax2.set_ylabel('Average Applications', fontsize=12, color='steelblue')
    ax2_twin.set_ylabel('Cumulative Damage (kg/ha)', fontsize=12, color='coral')
    ax2.set_title('Management Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([a.upper() for a in agents])
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    fig.suptitle('Pest Management Strategy Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Pest management strategies graph saved to: {output_file}")
    plt.close()


def plot_crop_state_variables(histories_dict, output_file='crop_state.png'):
    """
    Visualize DSSAT crop state variables throughout the growing season.

    Args:
        histories_dict: Dict mapping agent names to their episode histories
        output_file: Output filename for the plot
    """
    print(f"\nGenerating crop state variables graph...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = {'null': '#d62728', 'ppo': '#1f77b4', 'expert': '#2ca02c'}

    # Variables to plot with their labels and units
    variables = {
        'grnwt': ('Grain Weight', 'kg/ha', axes[0, 0]),
        'xlai': ('Leaf Area Index', 'LAI', axes[0, 1]),
        'canwaa': ('Canopy Weight', 'kg/ha', axes[1, 0]),
        'vstage': ('Vegetative Stage', 'Stage', axes[1, 1])
    }

    for var_name, (title, ylabel, ax) in variables.items():
        for agent_name, histories in histories_dict.items():
            if not histories:
                continue

            # Check if this variable was tracked
            if var_name not in histories[0] or not histories[0][var_name]:
                continue

            # Average across episodes, ensuring DAP and var data align
            max_len = max(len(h[var_name]) for h in histories if h.get(var_name))
            var_avg = []
            dap_vals = []

            for i in range(max_len):
                values = [h[var_name][i] for h in histories
                         if i < len(h.get(var_name, []))]
                if values:
                    var_avg.append(np.mean(values))
                    # Get corresponding DAP value (use first history with data at this index)
                    for h in histories:
                        if i < len(h.get('dap', [])):
                            dap_vals.append(h['dap'][i])
                            break
                    else:
                        dap_vals.append(i)  # Fallback to index

            if not var_avg:
                continue

            # Plot
            ax.plot(dap_vals, var_avg, linewidth=2.5,
                   color=colors.get(agent_name, '#999'),
                   label=agent_name.upper(), alpha=0.8)

        ax.set_xlabel('Days After Planting', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Crop Development Throughout Growing Season', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Crop state variables graph saved to: {output_file}")
    plt.close()


def plot_results(labels, returns):
    data_dict = {}
    for label, data in zip(labels, returns):
        data_dict[label] = data
    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df)
    ax.set_xlabel("policy")
    ax.set_ylabel("evaluation output")
    plt.savefig('pest_ppo_results.pdf')
    print("\nResults saved as 'pest_ppo_results.pdf'\n")
    plt.show()


def main():
    """Main training loop."""
    # Create environment with pests
    env_args = {
        'mode': 'all',
        'seed': 1024,
        'random_weather': True,
    }

    enable_pests = True
    pest_config = {
        'base_pressure': 0.35,
        'weather_sensitivity': 0.6,
        'damage_rate': 0.025,
        'pesticide_efficacy': 0.7,
        'pesticide_cost': 15.0,
    }

    env = GymDssatWrapperWithPests(
        gym.make('GymDssatPdi-v0', **env_args),
        enable_pests=enable_pests,
        pest_config=pest_config
    )

    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action keys: {env.action_keys}")

    # PPO training arguments
    ppo_args = {
        'gamma': 0.99,
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'seed': 123,
        'verbose': 1,
    }

    # Create and train agent
    ppo_agent = PPO('MlpPolicy', env, **ppo_args)

    print('Training PPO agent with pest management...')
    ppo_agent.learn(total_timesteps=1000000, progress_bar=True)
    print('Training done')

    ppo_agent.save("ppo_pest_model")
    print("Model saved as ppo_pest_model.zip")

    # Evaluate agents
    null_agent = NullAgent(env)
    expert_agent = ExpertAgent(env)

    print('Evaluating Null agent...')
    null_returns = evaluate(null_agent, n_episodes=100, enable_pests=enable_pests)
    print('Done')

    print('Evaluating PPO agent...')
    ppo_returns = evaluate(ppo_agent, n_episodes=100, enable_pests=enable_pests)
    print('Done')

    print('Evaluating Expert agent...')
    expert_returns = evaluate(expert_agent, n_episodes=100, enable_pests=enable_pests)
    print('Done')

    # Display results
    labels = ['null', 'ppo', 'expert']
    returns = [null_returns, ppo_returns, expert_returns]
    plot_results(labels, returns)

    with open("pest_eval_output.txt", 'w') as f:
        f.write("Null Agent: " + str(null_returns) + "\n")
        f.write("PPO Agent: " + str(ppo_returns) + "\n")
        f.write("Expert Agent: " + str(expert_returns) + "\n")

    # Generate pest-specific visualizations
    print('\n' + '='*70)
    print('GENERATING PEST ANALYSIS VISUALIZATIONS')
    print('='*70)

    print('\nTracking pest dynamics across episodes for visualization...')

    print('Tracking Null agent...')
    null_pest_histories = evaluate_with_pest_tracking(
        null_agent, 'null', n_episodes=3, enable_pests=enable_pests
    )

    print('Tracking PPO agent...')
    ppo_pest_histories = evaluate_with_pest_tracking(
        ppo_agent, 'ppo', n_episodes=3, enable_pests=enable_pests
    )

    print('Tracking Expert agent...')
    expert_pest_histories = evaluate_with_pest_tracking(
        expert_agent, 'expert', n_episodes=3, enable_pests=enable_pests
    )

    # Create visualization dictionaries
    histories_dict = {
        'null': null_pest_histories,
        'ppo': ppo_pest_histories,
        'expert': expert_pest_histories
    }

    # Generate the three main graphs
    plot_pest_impact(histories_dict, output_file='pest_impact.png')
    plot_pest_management_strategies(histories_dict, output_file='pest_management.png')
    plot_crop_state_variables(histories_dict, output_file='crop_state.png')

    print('\n' + '='*70)
    print('VISUALIZATION COMPLETE')
    print('='*70)
    print('\nGenerated files:')
    print('  1. pest_ppo_results.pdf - Overall policy comparison')
    print('  2. pest_impact.png - Pest impact over time (2 panels)')
    print('  3. pest_management.png - Management strategy comparison (2 panels)')
    print('  4. crop_state.png - Crop state variables (4 panels: grain, LAI, canopy, stage)')
    print('  5. pest_eval_output.txt - Numerical results')

    # Cleanup
    env.close()


if __name__ == '__main__':
    main()
