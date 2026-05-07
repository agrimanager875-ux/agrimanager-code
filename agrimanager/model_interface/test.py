from agrimanager.model_interface.model_factory import create_model

# Shared test messages for all tests
TEST_MESSAGES = [
    # Test 1: Simple user query (no system message)
    [
        {"role": "user", "content": "What is 2+2?"}
    ],

    # Test 2: System instruction + user query
    [
        {"role": "system", "content": "Please answer the following questions using Chinese."},
        {"role": "user", "content": "What is the capital of France?"}
    ],

    # Test 3: Multi-turn conversation (plus system message)
    [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is the Pythagorean theorem?"},
        {"role": "assistant", "content": "The Pythagorean theorem states that in a right triangle, a² + b² = c²."},
        {"role": "user", "content": "Can you give me an example?"}
    ],

    # Test 4: Consecutive user messages and model messages (edge case)
    [
        {"role": "user", "content": "I have a question."},
        {"role": "user", "content": "What is the speed of light?"},
        {"role": "assistant", "content": "The speed of light is 299,792,458 m/s."},
        {"role": "assistant", "content": "It's often rounded to 3×10⁸ m/s for simplicity."},
        {"role": "user", "content": "Thanks! That's helpful."},
    ],
]


def run_test(config_path: str, test_name: str):
    """
    Generic test function that runs the same tests for any model.

    Args:
        config_path: Path to the model configuration YAML file
        test_name: Name of the test (for display)
    """
    print("=" * 60)
    print(f"Testing: {test_name}")
    print("=" * 60)

    # Create model from config
    model_interface = create_model(config_path)

    # Generate responses
    print(f"\nGenerating {len(TEST_MESSAGES)} responses...")
    responses = model_interface.generate(TEST_MESSAGES)

    # Print results
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    for i, (messages, response) in enumerate(zip(TEST_MESSAGES, responses)):
        print(f"\n[Request {i+1}]")
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            print(f"  {role}: {content}")
        print(f"\n[Response {i+1}]")
        print(f"  {response}")
    print("\n" + "=" * 60 + "\n")


def test_openai_gpt4_1_mini():
    """Test OpenAI GPT-4.1-mini model."""
    config_path = "configs/openai/gpt-4.1-mini.yaml"
    run_test(config_path, "OpenAI GPT-4.1-mini")

def test_openai_gpt4_1_mini():
    """Test OpenAI GPT-4.1-mini model."""
    config_path = "configs/openai/gpt-4.1-mini.yaml"
    run_test(config_path, "OpenAI GPT-4.1-mini")

def test_google_gemini_2_0_flash():
    """Test Google Gemini 2.0 Flash model."""
    config_path = "configs/google/gemini-2.0-flash.yaml"
    run_test(config_path, "Google Gemini 2.0 Flash")


def test_vllm_offline_qwen2_5_3b_instruct():
    """Test vLLM Offline with Qwen2.5-3B-Instruct model."""
    config_path = "configs/vllm_offline/Qwen2.5-3B-Instruct.yaml"
    run_test(config_path, "vLLM Offline - Qwen2.5-3B-Instruct")

def test_vllm_offline_qwen2_5_7b_instruct():
    """Test vLLM Offline with Qwen2.5-7B-Instruct model."""
    config_path = "configs/vllm_offline/Qwen2.5-7B-Instruct.yaml"
    run_test(config_path, "vLLM Offline - Qwen2.5-7B-Instruct")

def test_vllm_server_qwen2_5_3b_instruct():
    """Test vLLM Server with Qwen2.5-3B-Instruct model."""
    config_path = "configs/vllm_server/Qwen2.5-3B-Instruct.yaml"
    run_test(config_path, "vLLM Server - Qwen2.5-3B-Instruct")

def test_vllm_server_gpt_oss_20b_medium():
    """Test vLLM Server with GPT-OSS-20B-Medium model."""
    config_path = "configs/vllm_server/gpt-oss-20b_medium.yaml"
    run_test(config_path, "vLLM Server - GPT-OSS-20B-Medium")


if __name__ == "__main__":
    # test_openai_gpt4_1_mini()
    # test_vllm_offline_qwen2_5_3b_instruct()
    # test_vllm_offline_qwen2_5_7b_instruct()
    # test_vllm_server_qwen2_5_3b_instruct()
    # test_google_gemini_2_0_flash()
    test_vllm_server_gpt_oss_20b_medium()