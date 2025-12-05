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
    "gemini-2.5-flash": { # Assuming same pricing for this hypothetical model
        "tier_1": {
            "input": 0.3,
            "output": 2.5
        },
        "tier_2": {
            "input": 0.3,
            "output": 2.5
        }
    },
    "gemini-2.5-flash-lite": { # Assuming same pricing for this hypothetical model
        "tier_1": {
            "input": 0.1,
            "output": 0.4
        },
        "tier_2": {
            "input": 0.1,
            "output": 0.4
        }
    }
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculates the estimated cost for a Gemini API call."""

    pricing_info = GEMINI_PRICING.get(model_name)
    if not pricing_info:
        return 0.0 # Return 0 if model not in pricing list

    if input_tokens <= TIER_1_TOKEN_LIMIT:
        tier = "tier_1"
    else:
        tier = "tier_2"

    input_price = pricing_info[tier]["input"]
    output_price = pricing_info[tier]["output"]

    input_cost = (input_tokens / TOKENS_PER_MILLION) * input_price
    output_cost = (output_tokens / TOKENS_PER_MILLION) * output_price

    total_cost = input_cost + output_cost
    return total_cost
