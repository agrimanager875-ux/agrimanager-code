import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from agrimanager.env.base.utils import create_environment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    print(f"Loading HF model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # ---- IMPORTANT FIX ----
    # Do NOT use device_map="" (requires accelerate)
    # Load model on CPU first, then .to(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.to(DEVICE)

    return tokenizer, model

def run_one_env(model, tokenizer, env_cfg):
    env, cfg = create_environment("gym_dssat", env_cfg)
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:
        prompt = f"Observation: {obs}\nAction:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        action_str = decoded.split("Action:")[-1].strip()

        try:
            action = float(action_str)
        except:
            action = 0.0

        obs, reward, done, info = env.step(action)
        total_reward += reward

    return total_reward

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/gym_dssat/maize_phase1/test.json")
    args = parser.parse_args()

    tokenizer, model = load_model(args.checkpoint)

    with open(args.dataset) as f:
        data = json.load(f)

    results = []

    for cfg in data:
        ret = run_one_env(model, tokenizer, cfg)
        results.append(ret)
        print(f"Return = {ret}")

    print("\nFINAL RESULTS")
    print(results)
    print("Avg:", sum(results) / len(results))

if __name__ == "__main__":
    main()
