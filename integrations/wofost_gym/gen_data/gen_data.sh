# TODO: NEED Test

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WOFOST_GYM_PATH="${SCRIPT_DIR}/../../../../../AgriManagerExternal/WOFOSTGym"

AGENT_TYPE="PPO"
RESULTS_DIR="${WOFOST_RESULTS_DIR:-${SCRIPT_DIR}/../results/wheat_agro_wso_delta}"
AGENT_DIR="${AGENT_DIR:-${RESULTS_DIR}/PPO/PPO_wheat_agro_wso_delta_mbs64_Gamma1_intvn10_5M}"
AGENT_PATH="${AGENT_DIR}/agent.pt"
SAVE_FOLDER="${AGENT_DIR}/data/"
INTVN_INTERVAL=10

cd "$WOFOST_GYM_PATH" && python3 -m data_generation.gen_data \
    --file-type npz \
    --save-folder $SAVE_FOLDER \
    --data-file test_data \
    --agent-type $AGENT_TYPE \
    --agent-path $AGENT_PATH \
    --npk.intvn-interval "$INTVN_INTERVAL" \
    --no-cuda
