# core/pricing.py

# Source: https://ai.google.dev/gemini-api/docs/pricing
# Rates for video and audio input
VIDEO_TOKENS_PER_SECOND = 263
AUDIO_TOKENS_PER_SECOND = 32
TOTAL_INPUT_TOKENS_PER_SECOND = VIDEO_TOKENS_PER_SECOND + AUDIO_TOKENS_PER_SECOND

# Pricing is per 1,000,000 tokens (1M)
TOKENS_PER_MILLION = 1_000_000

# Tiered pricing based on input prompt size
TIER_1_TOKEN_LIMIT = 200_000

# Prices in USD per 1M tokens
GEMINI_PRICING = {
    "gemini-3-pro-preview": {
        "tier_1": {
            "input": 2.00,
            "output": 12.00
        },
        "tier_2": {
            "input": 4.00,
            "output": 18.00
        }
    },
    "gemini-2.5-pro": {
        "tier_1": {
            "input": 1.25,
            "output": 10.00
        },
        "tier_2": {
            "input": 2.50,
            "output": 15.00
        }
    },
    "gemini-2.5-flash": {
        "input": 0.3,
        "output": 2.5
    },
    "gemini-2.5-flash-lite": {
        "input": 0.1,
        "output": 0.4
    },
    "qwen/qwen3-vl-8b-thinking": {
        "input": 0.18,
        "output": 2.1
    },
    "qwen/qwen3-vl-235b-a22b-thinking": {
        "input": 0.3,
        "output": 1.2
    },
    "openai/gpt-5.1": {
        "input": 1.25,
        "output": 10
    },
    "google/gemini-3-pro-preview": {
        "tier_1": {
            "input": 2.00,
            "output": 12.00
        },
        "tier_2": {
            "input": 4.00,
            "output": 18.00
        }
    },
    "google/gemini-2.5-pro": {
        "tier_1": {
            "input": 1.25,
            "output": 10.00
        },
        "tier_2": {
            "input": 2.50,
            "output": 15.00
        }
    },
}

#Video token per second
VIDEO_TOKENS = {
    "nvidia/nemotron-nano-12b-v2-vl:free": {
        "fps": 2,
        "token_per_frame": 256
    },
    "qwen/qwen3-vl-8b-thinking": {
        "fps": 2,
        "token_per_frame": 256
    },
    "qwen/qwen3-vl-235b-a22b-thinking": {
        "fps": 2,
        "token_per_frame": 256
    },
    "openai/gpt-5.1": {
        "fps": 2,
        "token_per_frame": 256
    }
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculates the estimated cost for a Gemini API call."""

    pricing_info = GEMINI_PRICING.get(model_name)
    if not pricing_info:
        return 0.0 # Return 0 if model not in pricing list

    if "tier_1" in pricing_info:
        if input_tokens <= TIER_1_TOKEN_LIMIT:
            tier = "tier_1"
        else:
            tier = "tier_2"

        input_price = pricing_info[tier]["input"]
        output_price = pricing_info[tier]["output"]
    else:
        input_price = pricing_info["input"]
        output_price = pricing_info["output"]

    input_cost = (input_tokens / TOKENS_PER_MILLION) * input_price
    output_cost = (output_tokens / TOKENS_PER_MILLION) * output_price

    total_cost = input_cost + output_cost
    return total_cost

def video_token_per_second(model_name: str):
    if "gemini" in model_name:
        return TOTAL_INPUT_TOKENS_PER_SECOND

    token_info = VIDEO_TOKENS.get(model_name)

    if not token_info:
        return 0.0
    fps = token_info["fps"]
    token_per_frame = token_info["token_per_frame"]

    return fps * token_per_frame
