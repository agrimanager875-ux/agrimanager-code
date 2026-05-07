# WOFOST Crop Traits Generator

This directory contains the utility used to regenerate the repository-managed
WOFOST crop traits.

## Output Location

Generated traits are written to:

`agrimanager/env/wofost_gym/crop_traits/`

That directory is the canonical in-repo location used by the `wofost_gym`
environment when `include_crop_traits: true`.

The generator emits:

- legacy flat 23D cards directly under `crop_traits/`
- schema-aware 23D cards under `crop_traits/traits_v1_23d/`
- schema-aware 6D cards under `crop_traits/traits_v1_6d/`

## Inputs

By default the generator reads from the external WOFOSTGym checkout:

- `env_config/crop/maize.yaml`
- `env_config/agro/maize_agro.yaml`

The root is resolved from `WOFOST_GYM_PATH` when set, otherwise it falls back
to `../AgriManagerExternal/WOFOSTGym` relative to this repository.

## Usage

From the repository root:

```bash
python integrations/wofost_gym/crop_pool_design/build_crop_traits.py
```

Generate only selected crops:

```bash
python integrations/wofost_gym/crop_pool_design/build_crop_traits.py --crops wheat maize barley
```

Override the output directory:

```bash
python integrations/wofost_gym/crop_pool_design/build_crop_traits.py --output-dir /tmp/crop_traits
```

## Notes

- By default the script regenerates every crop that has both crop and agro YAMLs.
- The script emits both `.json` structured traits and `.txt` prompt traits for both schemas.
