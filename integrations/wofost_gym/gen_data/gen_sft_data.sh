SCRIPT_DIR=$(dirname $(realpath $0))

MODEL_DIR="$SCRIPT_DIR/../results/wheat_agro_peak_wso/PPO/PPO_wheat_agro_peak_wso_mbs128"

ENV_NAME=wofost_gym
DATASET_ID=wheat
DATA_CONFIG="${DATA_CONFIG:?Set DATA_CONFIG to a legacy year-split config before running gen_sft_data.sh}"

python $SCRIPT_DIR/convert_llm_dataset.py \
    --data-path $MODEL_DIR/data/data.npz \
    --env-config $MODEL_DIR/config.yaml \
    --dataset-config $DATA_CONFIG \
    --output-dir $MODEL_DIR/data/ \
