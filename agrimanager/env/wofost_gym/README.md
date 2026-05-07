# wofost_gym Environment Wrapper

This module provides a wrapper around wofost_gym environments to make them compatible with the BaseEnv interface.

## Files

- `env_config.py` - Configuration class for wofost_gym environments
- `env.py` - Wrapper class that adapts wofost_gym to BaseEnv interface
- `simple_test.py` - Test script demonstrating basic usage
- `__init__.py` - Module exports

## Quick Start

### 1. Run the Simple Test

The easiest way to understand the environment is to run the test script:

```bash
cd "$AGRIMANAGER_ROOT"
python -m agrimanager.env.wofost_gym.simple_test
```

This will:
- Create a configuration
- Initialize the environment
- Show the system prompt for LLM agents
- Run several steps with random actions
- Display observations, rewards, and other key information

### 2. Basic Usage Example

```python
from agrimanager.env.wofost_gym import WOFOSTEnv, WOFOSTEnvConfig

# Create configuration
config = WOFOSTEnvConfig(
    env_id="lnpkw-v0",
    agro_file="wheat_agro.yaml",
    seed=42  # wofost_gym_path defaults to ../AgriManagerExternal/WOFOSTGym
)

# Create environment
env = WOFOSTEnv(config)

# Get system prompt (for LLM agents)
print(env.system_prompt())

# Reset environment
obs, info = env.reset()

# Run episode
done = False
total_reward = 0

while not done:
    action = env.env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")

# Cleanup
env.close()
```

## Configuration Options

### WOFOSTEnvConfig Parameters

- `env_id` (str): Environment template ID
  - Examples: `"lnpkw-v0"`, `"lnpk-v0"`, `"perennial-lnpkw-v0"`, `"multi-lnpkw-v0"`
  - Default: `"lnpkw-v0"`

- `agro_file` (str): Agromanagement YAML filename
  - Location: `{wofost_gym_path}/env_config/agro/`
  - Examples: `"wheat_agro.yaml"`, `"maize_agro.yaml"`
  - Default: `"wheat_agro.yaml"`

- `wofost_gym_path` (str): Path to wofost_gym installation
  - Default: the repository root's sibling `AgriManagerExternal/WOFOSTGym`
  - Override: set `WOFOST_GYM_PATH`

- `objective_id` (str): AgriManager management objective and reward definition
  - Examples: `"profit_max"`, `"yield_max"`, `"water_stewardship"`, `"nutrient_stewardship"`
  - Default: `"profit_max"`

- `env_reward` (str, optional): Legacy native WOFOST-Gym reward wrapper name
  - Examples: `"RewardFertilizationCostWrapper"`, `"RewardFertilizationThresholdWrapper"`
  - Default: `None`
  - New experiment configs should use `objective_id` instead.

- `wofost_params` (dict, optional): WOFOST model parameters to override
  - Examples: `{"NSOILBASE": 10.0, "TSUM1": 543.0}`
  - Default: `{}`

- `agro_params` (dict, optional): Agromanagement parameters to override
  - Examples: `{"latitude": 50.0, "longitude": 5.0}`
  - Default: `{}`

- `seed` (int, optional): Random seed for reproducibility
  - Default: `None`

- `include_crop_traits` (bool): Enable crop traits prompt injection
  - Default: `False`

- `crop_traits_dir` (str, optional): Root directory containing crop trait cards
  - Default: `agrimanager/env/wofost_gym/crop_traits`

- `trait_schema` (str, optional): Schema name used to select crop trait cards
  - Default: `traits_v1_23d`

## Crop Traits OOD Mode

Use this mode to train on seen crops with traits and evaluate zero-shot on unseen crop traits.

- Enable traits in environment config:
  - `include_crop_traits: true`
  - `trait_schema: traits_v1_23d` or `trait_schema: traits_v1_6d`
  - Maintained trait files live under `agrimanager/env/wofost_gym/crop_traits/`
  - Regeneration script lives under `integrations/wofost_gym/crop_pool_design/`

- Injection behavior:
  - The system prompt includes guidance to use crop traits as prior agronomic knowledge.
  - Every user turn prompt includes a `<crop traits>...</crop traits>` block before the observation.
  - Observation is wrapped in `<current observation>...</current observation>` tags.

- Fallback behavior:
  - If `include_crop_traits=false`, environment behavior is unchanged from legacy mode.
  - If `include_crop_traits=true`, the environment loads the schema-aware trait file selected by `trait_schema`.
  - If that file is missing, environment initialization raises an error immediately.

## Environment Templates

### Annual Crop Environments
- `lnpkw-v0` - Limited N, P, K, and Water (4 fertilizer + irrigation actions)
- `lnpk-v0` - Limited N, P, K only
- `lnw-v0` - Limited N and Water only
- `ln-v0` - Limited N only
- `lw-v0` - Limited Water only
- `pp-v0` - Potential Production (no limitations)

### Perennial Crop Environments
- `perennial-lnpkw-v0` - Multi-year crops with NPK and water management
- `perennial-lnpk-v0` - Multi-year crops with NPK management
- `grape-lnpkw-v0` - Grape-specific environment

### Multi-Farm Environments
- `multi-lnpkw-v0` - Manage multiple farms simultaneously

## Action Space

For standard `lnpkw-v0` environment:
- **Type**: Discrete(17)
- **Action 0**: Do nothing
- **Actions 1-4**: Apply N fertilizer (4 levels)
- **Actions 5-8**: Apply P fertilizer (4 levels)
- **Actions 9-12**: Apply K fertilizer (4 levels)
- **Actions 13-16**: Apply irrigation (4 levels)

## Observation Space

- **Type**: Box (continuous numpy array)
- **Contains**:
  - Crop state variables (development stage, biomass, etc.)
  - Soil nutrient levels (N, P, K available)
  - Soil moisture
  - Total nutrients and water applied
  - Weather forecast (temperature, radiation, rainfall)
  - Days elapsed since start

## Reward

- **Default**: Crop yield (WSO - Weight of Storage Organs)
- **Custom**: Can be modified with reward wrappers

## Advanced Usage

### Custom WOFOST Parameters

```python
config = WOFOSTEnvConfig(
    env_id="lnpkw-v0",
    agro_file="wheat_agro.yaml",
    wofost_params={
        "NSOILBASE": 10.0,      # Base N available (kg/ha)
        "TSUM1": 543.0,          # Temp sum to anthesis
        "TSUM2": 1194.0,         # Temp sum to maturity
        "SMFCF": 0.46,          # Field capacity
    },
    seed=42
)
```

### Legacy Native Reward Wrapper

```python
config = WOFOSTEnvConfig(
    env_id="lnpkw-v0",
    agro_file="wheat_agro.yaml",
    env_reward="RewardFertilizationCostWrapper",
    cost=0.5,  # Additional kwarg for the wrapper
    seed=42
)
```

### Different Crops and Locations

```python
config = WOFOSTEnvConfig(
    env_id="lnpkw-v0",
    agro_file="maize_agro.yaml",  # Different crop
    agro_params={
        "latitude": 40.0,          # Different location
        "longitude": -100.0,
        "year": 1990,              # Different year
    },
    seed=42
)
```

## System Prompt for LLM Agents

The environment provides a `system_prompt()` method that generates a description suitable for LLM-based agents:

```python
env = WOFOSTEnv(config)
prompt = env.system_prompt()
# Use this prompt to initialize your LLM agent
```

## Dependencies

- wofost_gym (installed via `pip install -e`)
- gymnasium
- numpy
- PyYAML

## Troubleshooting

### Import Error: "Cannot import utils"
- Make sure `wofost_gym_path` points to the wofost_gym repository root
- The `utils.py` file should exist at `{wofost_gym_path}/utils.py`

### Missing Configuration Files
- Check that YAML files exist in `{wofost_gym_path}/env_config/agro/`
- Make sure crop and site files are in their respective directories

### Environment Creation Fails
- Verify that wofost_gym is properly installed (`pip install -e pcse -e pcse_gym`)
- Check that all dependencies are installed
- Ensure the `env_id` is valid (registered in `pcse_gym/__init__.py`)

## Prompt Generation for LLM Agents

The module includes a `WOFOSTPromptGenerator` class for creating prompts for LLM-based agents:

```python
from agrimanager.env.wofost_gym import WOFOSTEnv, WOFOSTEnvConfig, WOFOSTPromptGenerator

# Create environment
config = WOFOSTEnvConfig(env_id="lnpkw-v0", agro_file="wheat_agro.yaml")
env = WOFOSTEnv(config)

# Create prompt generator from environment
prompt_gen = WOFOSTPromptGenerator.from_env(env.env)

# Get system prompt
system_prompt = prompt_gen.get_system_prompt()

# Get observation and generate turn prompt
obs, info = env.reset()
turn_prompt = prompt_gen.get_turn_prompt(obs)

print("System:", system_prompt)
print("\nUser:", turn_prompt)

# Simulate LLM response
action_id = 1  # Apply nitrogen fertilizer
action_desc = prompt_gen.describe_action(action_id)
print("\nAssistant:", action_desc)

# Parse LLM response back to action
parsed_action = prompt_gen.parse_action_response(action_desc)
print(f"\nParsed action ID: {parsed_action}")
```

### Manual Prompt Generator Configuration

```python
from agrimanager.env.wofost_gym import WOFOSTPromptGenerator

# Create prompt generator with custom settings
prompt_gen = WOFOSTPromptGenerator(
    crop_name="wheat",
    season_length=241,
    location="50.0°N, 5.0°E",
    num_fert=4,
    num_irrig=4,
    fert_amount=2.0,
    irrig_amount=0.5,
    output_vars=["FIN", "DVS", "WSO", "NAVAIL", "PAVAIL", "KAVAIL", "SM",
                 "TOTN", "TOTP", "TOTK", "TOTIRRIG", "IRRAD", "TEMP", "RAIN", "DAYS"]
)
```

## More Information

For more details about wofost_gym environments, see:
- [wofost_gym Repository](https://github.com/Intelligent-Reliable-Autonomous-Systems/wofost_gym)
- wofost_gym documentation in the repository
