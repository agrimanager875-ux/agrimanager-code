"""
Pest Management Simulation and Visualization

This script runs a full growing season simulation with pest management
and generates comprehensive graphs showing:
- Pest pressure over time
- Cumulative pest damage
- Pesticide applications
- Crop growth metrics
- Economic analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add agrimanager to path
sys.path.insert(0, os.path.dirname(__file__))

from agrimanager.env.gym_dssat.env import DSSATEnv
from agrimanager.env.gym_dssat.env_config import DSSATEnvConfig


class PestSimulation:
    """Run and visualize pest management simulations."""

    def __init__(self, config):
        self.config = config
        self.env = None
        self.history = {
            'dap': [],
            'pest_pressure': [],
            'pest_damage': [],
            'cumulative_damage': [],
            'days_since_pesticide': [],
            'pesticide_applications': [],
            'actions': [],
            'rewards': [],
            'xlai': [],
            'grnwt': [],
            'vstage': [],
            'swfac': [],
            'nstres': [],
            'tmax': [],
        }

    def create_environment(self):
        """Create the DSSAT environment with pest management."""
        print("Creating DSSAT environment with pest management...")
        self.env = DSSATEnv(self.config)
        print(f"✅ Environment created")
        print(f"   Output variables: {self.env.output_vars}")

    def simple_policy(self, obs_dict, step):
        """
        Simple rule-based pest management policy.

        Rules:
        1. Apply pesticide if pest_pressure > 0.6
        2. Apply fertilizer at growth stages (days 30, 50, 70)
        3. Irrigate if water stress > 0.5
        """
        dap = obs_dict.get('dap', 0)
        pest_pressure = obs_dict.get('pest_pressure', 0)
        swfac = obs_dict.get('swfac', 0)
        days_since_pest = obs_dict.get('days_since_pesticide', 999)

        # Priority 1: Pesticide if pressure is high and enough time has passed
        if pest_pressure > 0.6 and days_since_pest > 7:
            return "<answer>Apply pesticide.<answer>"

        # Priority 2: Fertilizer at key growth stages
        if dap in [30, 50, 70]:
            return "<answer>Apply 25 kg/ha nitrogen fertilizer.<answer>"

        # Priority 3: Irrigation if water stressed
        if swfac > 0.5:
            return "<answer>Irrigate with 15 mm of water.<answer>"

        # Default: do nothing
        return "<answer>Do nothing.<answer>"

    def aggressive_pesticide_policy(self, obs_dict, step):
        """Aggressive policy: spray pesticides frequently."""
        dap = obs_dict.get('dap', 0)
        pest_pressure = obs_dict.get('pest_pressure', 0)
        days_since_pest = obs_dict.get('days_since_pesticide', 999)

        # Spray every 10 days if pressure > 0.4
        if pest_pressure > 0.4 and days_since_pest > 10:
            return "<answer>Apply pesticide.<answer>"

        # Fertilizer at standard times
        if dap in [30, 50, 70]:
            return "<answer>Apply 25 kg/ha nitrogen fertilizer.<answer>"

        return "<answer>Do nothing.<answer>"

    def no_pesticide_policy(self, obs_dict, step):
        """Control policy: never use pesticides."""
        dap = obs_dict.get('dap', 0)

        # Only fertilizer
        if dap in [30, 50, 70]:
            return "<answer>Apply 25 kg/ha nitrogen fertilizer.<answer>"

        return "<answer>Do nothing.<answer>"

    def parse_observation(self, obs):
        """Parse observation into dictionary."""
        obs_dict = {}
        if isinstance(obs, dict):
            return obs
        elif isinstance(obs, (list, np.ndarray)):
            for i, var in enumerate(self.env.output_vars):
                if i < len(obs):
                    obs_dict[var] = obs[i]
            return obs_dict
        else:
            # It's a string prompt, extract from environment state
            obs_dict = {
                'dap': self.env._current_dap,
                'pest_pressure': getattr(self.env, '_pest_pressure', 0),
                'pest_damage': getattr(self.env, '_cumulative_pest_damage', 0),
                'days_since_pesticide': getattr(self.env, '_days_since_pesticide', 999),
            }
            return obs_dict

    def run_simulation(self, policy_fn, max_steps=200):
        """Run a full simulation using the given policy."""
        print(f"\nRunning simulation with policy: {policy_fn.__name__}")

        # Reset history
        for key in self.history.keys():
            self.history[key] = []

        # Reset environment
        obs, info = self.env.reset()
        obs_dict = self.parse_observation(obs)
        done = False
        step = 0

        while not done and step < max_steps:
            # Record current state
            self.history['dap'].append(obs_dict.get('dap', step))
            self.history['pest_pressure'].append(getattr(self.env, '_pest_pressure', 0))
            self.history['pest_damage'].append(getattr(self.env, '_cumulative_pest_damage', 0))
            self.history['days_since_pesticide'].append(getattr(self.env, '_days_since_pesticide', 999))
            self.history['xlai'].append(obs_dict.get('xlai', 0))
            self.history['grnwt'].append(obs_dict.get('grnwt', 0))
            self.history['vstage'].append(obs_dict.get('vstage', 0))
            self.history['swfac'].append(obs_dict.get('swfac', 0))
            self.history['nstres'].append(obs_dict.get('nstres', 0))
            self.history['tmax'].append(obs_dict.get('tmax', 25))

            # Get action from policy
            action = policy_fn(obs_dict, step)
            self.history['actions'].append(action)

            # Track pesticide applications
            if 'pesticide' in action.lower():
                self.history['pesticide_applications'].append(obs_dict.get('dap', step))

            # Execute action
            obs, reward, done, info = self.env.step(action)
            obs_dict = self.parse_observation(obs)

            self.history['rewards'].append(reward)

            step += 1

            if step % 20 == 0:
                print(f"  Step {step}/{max_steps}: DAP={obs_dict.get('dap', step)}, "
                      f"Pest={getattr(self.env, '_pest_pressure', 0):.3f}, "
                      f"Damage={getattr(self.env, '_cumulative_pest_damage', 0):.2f}")

        print(f"✅ Simulation complete: {step} steps")
        print(f"   Total pest damage: {self.history['pest_damage'][-1]:.2f} kg/ha")
        print(f"   Pesticide applications: {len(self.history['pesticide_applications'])}")
        print(f"   Final grain weight: {self.history['grnwt'][-1]:.2f} kg/ha")
        print(f"   Total reward: {sum(self.history['rewards']):.2f}")

        return self.history


def plot_pest_simulation_results(histories, policy_names, output_path='pest_simulation_results.png'):
    """
    Create comprehensive visualization of pest simulation results.

    Args:
        histories: List of history dictionaries from different policies
        policy_names: List of policy names
        output_path: Path to save the figure
    """
    print(f"\nGenerating visualization...")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # ========================================================================
    # 1. Pest Pressure Over Time
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax1.plot(hist['dap'], hist['pest_pressure'], label=name,
                linewidth=2, color=colors[i % len(colors)])
        # Mark pesticide applications
        for app_day in hist['pesticide_applications']:
            if app_day in hist['dap']:
                idx = hist['dap'].index(app_day)
                ax1.scatter(app_day, hist['pest_pressure'][idx],
                           marker='v', s=100, color=colors[i % len(colors)],
                           edgecolors='black', zorder=5)

    ax1.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='High pressure threshold')
    ax1.set_xlabel('Days After Planting')
    ax1.set_ylabel('Pest Pressure (0-1)')
    ax1.set_title('Pest Pressure Dynamics')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # 2. Cumulative Pest Damage
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax2.plot(hist['dap'], hist['pest_damage'], label=name,
                linewidth=2, color=colors[i % len(colors)])

    ax2.set_xlabel('Days After Planting')
    ax2.set_ylabel('Cumulative Damage (kg/ha)')
    ax2.set_title('Cumulative Pest Damage')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # 3. Days Since Pesticide Application
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        days_capped = [min(d, 30) for d in hist['days_since_pesticide']]
        ax3.plot(hist['dap'], days_capped, label=name,
                linewidth=2, color=colors[i % len(colors)])

    ax3.axhline(y=14, color='orange', linestyle='--', alpha=0.5,
               label='Pesticide efficacy duration')
    ax3.set_xlabel('Days After Planting')
    ax3.set_ylabel('Days Since Pesticide (capped at 30)')
    ax3.set_title('Pesticide Application Timing')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # 4. Crop Growth - Grain Weight
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax4.plot(hist['dap'], hist['grnwt'], label=name,
                linewidth=2, color=colors[i % len(colors)])

    ax4.set_xlabel('Days After Planting')
    ax4.set_ylabel('Grain Weight (kg/ha)')
    ax4.set_title('Grain Development')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # 5. Crop Growth - Leaf Area Index
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax5.plot(hist['dap'], hist['xlai'], label=name,
                linewidth=2, color=colors[i % len(colors)])

    ax5.set_xlabel('Days After Planting')
    ax5.set_ylabel('Leaf Area Index (m²/m²)')
    ax5.set_title('Canopy Development')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ========================================================================
    # 6. Vegetative Stage
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax6.plot(hist['dap'], hist['vstage'], label=name,
                linewidth=2, color=colors[i % len(colors)])

    ax6.set_xlabel('Days After Planting')
    ax6.set_ylabel('Vegetative Stage (# leaves)')
    ax6.set_title('Crop Development Stage')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ========================================================================
    # 7. Water Stress
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax7.plot(hist['dap'], hist['swfac'], label=name,
                linewidth=2, color=colors[i % len(colors)])

    ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Stress threshold')
    ax7.set_xlabel('Days After Planting')
    ax7.set_ylabel('Water Stress Factor (0-1)')
    ax7.set_title('Water Stress')
    ax7.legend(loc='best', fontsize=8)
    ax7.grid(True, alpha=0.3)

    # ========================================================================
    # 8. Nitrogen Stress
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax8.plot(hist['dap'], hist['nstres'], label=name,
                linewidth=2, color=colors[i % len(colors)])

    ax8.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Stress threshold')
    ax8.set_xlabel('Days After Planting')
    ax8.set_ylabel('Nitrogen Stress Factor (0-1)')
    ax8.set_title('Nitrogen Stress')
    ax8.legend(loc='best', fontsize=8)
    ax8.grid(True, alpha=0.3)

    # ========================================================================
    # 9. Temperature
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax9.plot(hist['dap'], hist['tmax'], label=name,
                linewidth=1, alpha=0.7, color=colors[i % len(colors)])

    ax9.set_xlabel('Days After Planting')
    ax9.set_ylabel('Max Temperature (°C)')
    ax9.set_title('Daily Maximum Temperature')
    ax9.legend(loc='best', fontsize=8)
    ax9.grid(True, alpha=0.3)

    # ========================================================================
    # 10. Economic Summary - Bar Chart
    # ========================================================================
    ax10 = fig.add_subplot(gs[3, :2])

    metrics = []
    for hist, name in zip(histories, policy_names):
        final_yield = hist['grnwt'][-1] if hist['grnwt'] else 0
        total_damage = hist['pest_damage'][-1] if hist['pest_damage'] else 0
        num_pesticides = len(hist['pesticide_applications'])

        metrics.append({
            'name': name,
            'yield': final_yield,
            'damage': total_damage,
            'pesticides': num_pesticides,
            'total_reward': sum(hist['rewards']) if hist['rewards'] else 0
        })

    x = np.arange(len(metrics))
    width = 0.2

    yields = [m['yield'] for m in metrics]
    damages = [m['damage'] for m in metrics]
    pesticides = [m['pesticides'] * 100 for m in metrics]  # Scale for visibility

    ax10.bar(x - width, yields, width, label='Final Yield (kg/ha)', color='green', alpha=0.7)
    ax10.bar(x, damages, width, label='Total Damage (kg/ha)', color='red', alpha=0.7)
    ax10.bar(x + width, pesticides, width, label='Pesticides × 100', color='blue', alpha=0.7)

    ax10.set_xlabel('Policy')
    ax10.set_ylabel('Value')
    ax10.set_title('Economic Summary by Policy')
    ax10.set_xticks(x)
    ax10.set_xticklabels([m['name'] for m in metrics], rotation=15, ha='right')
    ax10.legend()
    ax10.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # 11. Summary Statistics Table
    # ========================================================================
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')

    table_data = [['Policy', 'Yield\n(kg/ha)', 'Damage\n(kg/ha)', '# Pest.\nApps', 'Total\nReward']]
    for m in metrics:
        table_data.append([
            m['name'][:15],
            f"{m['yield']:.0f}",
            f"{m['damage']:.1f}",
            f"{m['pesticides']}",
            f"{m['total_reward']:.0f}"
        ])

    table = ax11.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    # Overall title
    fig.suptitle('Pest Management Simulation Results - DSSAT AgriManager',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")

    return fig


def main():
    """Main execution function."""
    print("=" * 80)
    print("PEST MANAGEMENT SIMULATION AND VISUALIZATION")
    print("=" * 80)

    # Configuration
    config = DSSATEnvConfig(
        env_id="maize-pest-simulation",
        llm_mode=True,
        enable_pests=True,
        pest_config={
            "base_pressure": 0.35,
            "weather_sensitivity": 0.6,
            "damage_rate": 0.025,
            "pesticide_efficacy": 0.7,
            "pesticide_cost": 15.0,
        },
        turn_num=200,
    )

    # Create simulation
    sim = PestSimulation(config)
    sim.create_environment()

    # Run multiple policies
    policies = [
        (sim.simple_policy, "Smart IPM"),
        (sim.aggressive_pesticide_policy, "Aggressive Spray"),
        (sim.no_pesticide_policy, "No Pesticide"),
    ]

    histories = []
    policy_names = []

    for policy_fn, name in policies:
        history = sim.run_simulation(policy_fn, max_steps=200)
        histories.append(history)
        policy_names.append(name)

        # Reset environment for next policy
        if policy_fn != policies[-1][0]:  # Don't reset after last policy
            sim.env.close()
            sim.env = DSSATEnv(config)

    # Generate visualization
    output_file = f"pest_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_pest_simulation_results(histories, policy_names, output_file)

    # Clean up
    sim.env.close()

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"📊 Results saved to: {output_file}")
    print("\n✅ Compare different pest management strategies:")
    for name, hist in zip(policy_names, histories):
        print(f"\n{name}:")
        print(f"  Final Yield: {hist['grnwt'][-1]:.0f} kg/ha")
        print(f"  Pest Damage: {hist['pest_damage'][-1]:.1f} kg/ha")
        print(f"  Pesticide Apps: {len(hist['pesticide_applications'])}")
        print(f"  Total Reward: {sum(hist['rewards']):.0f}")


if __name__ == "__main__":
    main()
