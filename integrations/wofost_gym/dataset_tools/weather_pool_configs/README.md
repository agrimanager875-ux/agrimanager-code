Weather-pool generation configs live here.

These YAML files are inputs to `integrations/wofost_gym/dataset_tools/build_weather_pool.py`.
They define how to generate and validate raw `(crop, year, latitude, longitude)`
scenario pools. They are not final dataset configs.

Use `entrypoints/dataset/examples/wofost/` or
`experiments/t1_1_weather_regime_shift/config/` for artifact-first dataset
examples.

After building a reusable weather pool, bundle the required NASA POWER cache
files with:

```bash
python integrations/wofost_gym/dataset_tools/package_weather_pool.py \
  --pool-dir integrations/wofost_gym/dataset_tools/weather_pool_20crop_3200_val128_test512 \
  --year-padding 1 \
  --clean
```

This writes both:

- `meteo_cache/` for local runtime use
- `meteo_cache.tar.gz` for upload/distribution

Upload the prepared pool with:

```bash
python integrations/wofost_gym/dataset_tools/upload_weather_pool.py \
  --pool-dir integrations/wofost_gym/dataset_tools/weather_pool_20crop_3200_val128_test512
```
