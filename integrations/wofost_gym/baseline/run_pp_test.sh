#!/bin/bash
set -euo pipefail

CROPS=(potato barley millet sorghum sunflower)

for crop in "${CROPS[@]}"; do
    dataset="${crop}_europe_interval_10"
    test_file="data/wofost_gym/${dataset}/test.parquet"
    output_dir="results/wofost_gym/${dataset}/test/pp_v0_baseline"

    echo "============================================================"
    echo "Running pp-v0 baseline for: ${crop}"
    echo "test-file: ${test_file}"
    echo "output-dir: ${output_dir}"
    echo "============================================================"

    bash integrations/wofost_gym/baseline/pp_test.sh \
        --test-files "${test_file}" \
        --output-dir "${output_dir}" \
        --intvn-interval 10
done
