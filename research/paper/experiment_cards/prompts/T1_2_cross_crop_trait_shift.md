# T1.2 Cross-Crop Trait Shift Prompts

T1.2 changes crop identity and optionally exposes crop traits. The main prompt question is whether explicit trait context helps the policy generalize from ID crops to held-out crops.

## Covered Variants

| Variant | Prompt change |
| --- | --- |
| coverage: `4ID`, `8ID`, `16ID` | source crop set changes. |
| validation group: ID crops vs held-out crops | crop identity and crop state values change. |
| traits enabled/disabled | optional `<crop traits>` block appears or is omitted. |
| think vs no-think | response-format block changes. |

## Stable Prompt Contract

| Block | T1.2 behavior |
| --- | --- |
| System prompt | common agricultural-management system prompt. |
| Objective | `profit_max`. |
| Crop identity | always shown in the episode context. |
| Crop traits | controlled by `env.include_crop_traits`. |
| Observation schema | native WOFOST observation fields. |
| Action menu | WOFOST action menu configured by the experiment YAML. |

## Representative Trait Block

```text
<crop traits>
- Crop: maize.
- Maturity class: medium.
- Water stress sensitivity: moderate.
- Nutrient demand: high.
- Typical harvest organ: storage organ dry matter / WSO.
</crop traits>
```

Trait text should be treated as model-facing context, not as a separate source dataset. Trait ablations should differ only in this block; the source pool, split construction, seed, and evaluation labels should otherwise match.

## Representative User Prompt Skeleton

```text
We are growing maize from sow to maturity. The planned growing window spans
around 120 days, actions are taken every 10 days, and today is day 30 of the
season.

<management objective id="profit_max">
Maximize terminal profit after accounting for yield revenue and management
input costs.
</management objective>

<crop traits>
- Crop: maize.
- Maturity class: medium.
- Water stress sensitivity: moderate.
- Nutrient demand: high.
</crop traits>

<current observation>
DVS=0.35, WSO=120.0 kg/ha, SM=0.29, RAIN=0.0 cm, NAVAIL=35.0 kg/ha,
PAVAIL=12.0 kg/ha, KAVAIL=45.0 kg/ha, TOTN=0.0 kg/ha, TOTP=0.0 kg/ha,
TOTK=0.0 kg/ha, TOTIRRIG=0.0 cm.
</current observation>

Available actions (pick exactly one):
0. Take no action.
1. Irrigate with 0.5 cm of water.
2. Apply 20.0 kg/ha nitrogen fertilizer.
```

## Documentation Boundary

The old capture file repeated full prompts for every coverage setting, held-out split, and response mode. For documentation, keep the trait contract and one representative prompt skeleton; regenerate exact prompts when testing a specific config.
