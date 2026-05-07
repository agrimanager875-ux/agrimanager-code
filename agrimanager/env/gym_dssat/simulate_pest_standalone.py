"""
Standalone Pest Management Simulation

This version simulates pest dynamics without requiring DSSAT,
allowing you to visualize and test pest management strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class StandalonePestSimulator:
    """Simulate pest dynamics independently of DSSAT."""

    def __init__(self, config):
        self.config = config
        self.pest_pressure = config.get('base_pressure', 0.35)
        self.cumulative_damage = 0.0
        self.days_since_pesticide = 999
        self.pesticide_applications = 0

        # Crop growth simulation (simplified)
        self.dap = 0
        self.grain_weight = 0.0
        self.lai = 0.0
        self.vstage = 0.0

        # Weather (simplified patterns)
        np.random.seed(42)
        self.temperatures = 20 + 8 * np.sin(np.linspace(0, 2*np.pi, 200)) + np.random.normal(0, 2, 200)
        self.srad = 18 + 6 * np.sin(np.linspace(0, 2*np.pi, 200)) + np.random.normal(0, 1.5, 200)

    def reset(self):
        """Reset simulation."""
        self.pest_pressure = self.config.get('base_pressure', 0.35)
        self.cumulative_damage = 0.0
        self.days_since_pesticide = 999
        self.pesticide_applications = 0
        self.dap = 0
        self.grain_weight = 0.0
        self.lai = 0.0
        self.vstage = 0.0

    def apply_pesticide(self):
        """Apply pesticide."""
        efficacy = self.config.get('pesticide_efficacy', 0.7)
        self.pest_pressure *= (1 - efficacy)
        self.days_since_pesticide = 0
        self.pesticide_applications += 1

    def step(self, apply_pesticide=False):
        """Advance one day."""
        self.dap += 1

        # Get weather
        idx = min(self.dap - 1, len(self.temperatures) - 1)
        tmax = self.temperatures[idx]
        srad = self.srad[idx]

        # Apply pesticide if requested
        if apply_pesticide:
            self.apply_pesticide()

        # Update pest pressure
        self.days_since_pesticide += 1
        weather_sensitivity = self.config.get('weather_sensitivity', 0.6)
        temp_factor = max(0, (tmax - 20) / 15)
        light_factor = max(0, (srad - 15) / 10)

        growth_rate = 0.05 * (1 + weather_sensitivity * (temp_factor + light_factor) / 2)

        # Pesticide decay
        decay = 0.0
        if self.days_since_pesticide < 14:
            decay = 0.02 * (14 - self.days_since_pesticide) / 14

        self.pest_pressure += growth_rate - decay
        self.pest_pressure = max(0.0, min(1.0, self.pest_pressure))

        # Accumulate damage
        damage_rate = self.config.get('damage_rate', 0.025)
        daily_damage = damage_rate * self.pest_pressure
        self.cumulative_damage += daily_damage

        # Simple crop growth model
        if self.dap < 40:
            self.vstage = self.dap / 4.0
            self.lai = self.dap / 20.0
        elif self.dap < 80:
            self.vstage = 10 + (self.dap - 40) / 10.0
            self.lai = 2.0 + (self.dap - 40) / 15.0
        else:
            self.vstage = 14.0
            self.lai = max(0, 5.0 - (self.dap - 80) / 30.0)

        # Grain weight (logistic growth, reduced by pests)
        if self.dap > 60:
            max_yield = 8500 - self.cumulative_damage * 100  # Damage reduces yield
            self.grain_weight = max_yield / (1 + np.exp(-(self.dap - 120) / 15))

        return {
            'dap': self.dap,
            'pest_pressure': self.pest_pressure,
            'pest_damage': self.cumulative_damage,
            'days_since_pesticide': self.days_since_pesticide,
            'grnwt': self.grain_weight,
            'xlai': self.lai,
            'vstage': self.vstage,
            'tmax': tmax,
            'srad': srad,
            'swfac': 0.2 + 0.3 * np.sin(self.dap / 10),  # Simulated water stress
            'nstres': 0.15 + 0.25 * np.sin(self.dap / 15),  # Simulated N stress
        }


def run_simulation(config, policy_fn, policy_name, max_steps=150):
    """Run simulation with a policy."""
    print(f"\nRunning: {policy_name}")

    sim = StandalonePestSimulator(config)
    sim.reset()

    history = {
        'dap': [], 'pest_pressure': [], 'pest_damage': [],
        'days_since_pesticide': [], 'pesticide_applications': [],
        'grnwt': [], 'xlai': [], 'vstage': [], 'tmax': [],
        'srad': [], 'swfac': [], 'nstres': []
    }

    for step in range(max_steps):
        # Get current state
        state = sim.step(apply_pesticide=False)  # Don't apply yet

        # Policy decision
        apply_pest = policy_fn(state, step)

        # Apply if decided
        if apply_pest:
            state = sim.step(apply_pesticide=True)
            history['pesticide_applications'].append(state['dap'])
        else:
            # Already stepped above, just record
            pass

        # Record history
        for key in history.keys():
            if key == 'pesticide_applications':
                continue
            history[key].append(state[key])

        if step % 30 == 0:
            print(f"  Day {state['dap']:3d}: Pest={state['pest_pressure']:.3f}, "
                  f"Damage={state['pest_damage']:.2f}, Apps={sim.pesticide_applications}")

    final_yield = history['grnwt'][-1]
    total_damage = history['pest_damage'][-1]
    num_apps = sim.pesticide_applications
    pesticide_cost = num_apps * config.get('pesticide_cost', 15.0)

    print(f"  ✅ Final Yield: {final_yield:.0f} kg/ha")
    print(f"  📉 Total Damage: {total_damage:.2f} kg/ha")
    print(f"  🪲 Pesticide Apps: {num_apps}")
    print(f"  💰 Pesticide Cost: ${pesticide_cost:.2f}")

    return history


def plot_results(histories, policy_names, output_file='pest_simulation_results.png'):
    """Generate comprehensive visualization."""
    print(f"\n📊 Generating visualization...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Pest Pressure
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax1.plot(hist['dap'], hist['pest_pressure'], label=name, linewidth=2, color=colors[i])
        for app_day in hist['pesticide_applications']:
            idx = hist['dap'].index(app_day) if app_day in hist['dap'] else None
            if idx:
                ax1.scatter(app_day, hist['pest_pressure'][idx], marker='v', s=100,
                           color=colors[i], edgecolors='black', zorder=5)
    ax1.axhline(0.6, color='red', linestyle='--', alpha=0.5, label='Action threshold')
    ax1.set_xlabel('Days After Planting')
    ax1.set_ylabel('Pest Pressure (0-1)')
    ax1.set_title('Pest Pressure Dynamics')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative Damage
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax2.plot(hist['dap'], hist['pest_damage'], label=name, linewidth=2, color=colors[i])
    ax2.set_xlabel('Days After Planting')
    ax2.set_ylabel('Cumulative Damage (kg/ha)')
    ax2.set_title('Cumulative Pest Damage')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Days Since Pesticide
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        days_capped = [min(d, 30) for d in hist['days_since_pesticide']]
        ax3.plot(hist['dap'], days_capped, label=name, linewidth=2, color=colors[i])
    ax3.axhline(14, color='orange', linestyle='--', alpha=0.5, label='Efficacy duration')
    ax3.set_xlabel('Days After Planting')
    ax3.set_ylabel('Days Since Pesticide (max 30)')
    ax3.set_title('Pesticide Application Timing')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Grain Weight
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax4.plot(hist['dap'], hist['grnwt'], label=name, linewidth=2, color=colors[i])
    ax4.set_xlabel('Days After Planting')
    ax4.set_ylabel('Grain Weight (kg/ha)')
    ax4.set_title('Grain Development')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. LAI
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax5.plot(hist['dap'], hist['xlai'], label=name, linewidth=2, color=colors[i])
    ax5.set_xlabel('Days After Planting')
    ax5.set_ylabel('Leaf Area Index')
    ax5.set_title('Canopy Development')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Vegetative Stage
    ax6 = fig.add_subplot(gs[1, 2])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax6.plot(hist['dap'], hist['vstage'], label=name, linewidth=2, color=colors[i])
    ax6.set_xlabel('Days After Planting')
    ax6.set_ylabel('Vegetative Stage')
    ax6.set_title('Crop Development Stage')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. Water Stress
    ax7 = fig.add_subplot(gs[2, 0])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax7.plot(hist['dap'], hist['swfac'], label=name, linewidth=2, color=colors[i])
    ax7.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Stress threshold')
    ax7.set_xlabel('Days After Planting')
    ax7.set_ylabel('Water Stress Factor')
    ax7.set_title('Water Stress (Simulated)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. Nitrogen Stress
    ax8 = fig.add_subplot(gs[2, 1])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax8.plot(hist['dap'], hist['nstres'], label=name, linewidth=2, color=colors[i])
    ax8.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Stress threshold')
    ax8.set_xlabel('Days After Planting')
    ax8.set_ylabel('Nitrogen Stress Factor')
    ax8.set_title('Nitrogen Stress (Simulated)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # 9. Temperature
    ax9 = fig.add_subplot(gs[2, 2])
    for i, (hist, name) in enumerate(zip(histories, policy_names)):
        ax9.plot(hist['dap'], hist['tmax'], label=name, linewidth=1, alpha=0.7, color=colors[i])
    ax9.set_xlabel('Days After Planting')
    ax9.set_ylabel('Max Temperature (°C)')
    ax9.set_title('Daily Maximum Temperature')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # 10. Economic Bar Chart
    ax10 = fig.add_subplot(gs[3, :2])
    metrics = []
    for hist, name in zip(histories, policy_names):
        metrics.append({
            'name': name,
            'yield': hist['grnwt'][-1],
            'damage': hist['pest_damage'][-1],
            'pesticides': len(hist['pesticide_applications'])
        })

    x = np.arange(len(metrics))
    width = 0.25
    yields = [m['yield'] for m in metrics]
    damages = [m['damage'] * 100 for m in metrics]  # Scale for visibility
    pesticides = [m['pesticides'] * 200 for m in metrics]  # Scale for visibility

    ax10.bar(x - width, yields, width, label='Final Yield (kg/ha)', color='green', alpha=0.7)
    ax10.bar(x, damages, width, label='Damage × 100', color='red', alpha=0.7)
    ax10.bar(x + width, pesticides, width, label='Pesticides × 200', color='blue', alpha=0.7)
    ax10.set_xlabel('Policy')
    ax10.set_ylabel('Value')
    ax10.set_title('Economic Summary by Policy')
    ax10.set_xticks(x)
    ax10.set_xticklabels([m['name'] for m in metrics], rotation=15, ha='right')
    ax10.legend()
    ax10.grid(True, alpha=0.3, axis='y')

    # 11. Summary Table
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')

    table_data = [['Policy', 'Yield\n(kg/ha)', 'Damage\n(kg/ha)', '# Apps']]
    for m in metrics:
        table_data.append([
            m['name'][:15],
            f"{m['yield']:.0f}",
            f"{m['damage']:.2f}",
            f"{m['pesticides']}"
        ])

    table = ax11.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.35, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    fig.suptitle('Pest Management Simulation Results - AgriManager (Standalone)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_file}")
    plt.close()


def main():
    """Main execution."""
    print("=" * 80)
    print("PEST MANAGEMENT STANDALONE SIMULATION")
    print("=" * 80)

    config = {
        'base_pressure': 0.35,
        'weather_sensitivity': 0.6,
        'damage_rate': 0.025,
        'pesticide_efficacy': 0.7,
        'pesticide_cost': 15.0,
    }

    # Define policies
    def smart_ipm(state, step):
        return state['pest_pressure'] > 0.6 and state['days_since_pesticide'] > 7

    def aggressive(state, step):
        return state['pest_pressure'] > 0.4 and state['days_since_pesticide'] > 10

    def no_pesticide(state, step):
        return False

    policies = [
        (smart_ipm, "Smart IPM"),
        (aggressive, "Aggressive Spray"),
        (no_pesticide, "No Pesticide"),
    ]

    # Run simulations
    histories = []
    policy_names = []

    for policy_fn, name in policies:
        history = run_simulation(config, policy_fn, name, max_steps=150)
        histories.append(history)
        policy_names.append(name)

    # Generate visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'pest_simulation_{timestamp}.png'
    plot_results(histories, policy_names, output_file)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Graph saved to: {output_file}")
    print("\n✅ Compare strategies:")
    for name, hist in zip(policy_names, histories):
        print(f"\n{name}:")
        print(f"  Yield: {hist['grnwt'][-1]:.0f} kg/ha")
        print(f"  Damage: {hist['pest_damage'][-1]:.2f} kg/ha")
        print(f"  Apps: {len(hist['pesticide_applications'])}")


if __name__ == "__main__":
    main()
