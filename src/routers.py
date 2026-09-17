import json

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

router_tokenizer = AutoTokenizer.from_pretrained("models/intent_classifier/models/intent_classifier")
router_model = AutoModelForSequenceClassification.from_pretrained("models/intent_classifier/models/intent_classifier")
router_model.eval()

RULES = {
    "escalate": ["data loss", "security breach", "production down", "corruption"],
    "qa": ["according to the docs", "documentation", "what does the manual say"],
}

INTENT_TO_ROUTE = {
    "authentication": "support_specialist",
    "network": "tools",
    "deployment": "tools",
    "database": "tools",
    "gpu": "tools",
    "api": "support_specialist",
    "package": "tools",
    "general": "support_specialist",
}

def rule_first(text: str):
    lowered = text.lower()
    for route, phrases in RULES.items():
        if any(p in lowered for p in phrases):
            return {"route": route, "source": "rule", "confidence": 1.0}
    return None

@torch.no_grad()
def classifier_route(text: str):
    inputs = router_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    logits = router_model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs))
    
    return {
        "intent": router_model.config.id2label[idx],
        "confidence": float(probs[idx]),
        "source": "classifier",
    }

def baseline_router(text: str):
    hard_rule = rule_first(text)
    if hard_rule:
        return hard_rule
        
    pred = classifier_route(text)
    # TODO(student 5): complete confidence policy and intent→specialist mapping.
    route = INTENT_TO_ROUTE.get(pred["intent"], "support_specialist")

    pred.update({"route": route})
    return pred


ALLOWED_ROUTES = {
    "qa",
    "support_specialist",
    "tools",
    "escalate",
}

LLM_ROUTER_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LLM_ROUTER_ADAPTER_PATH = "models/LoRA/models/support_adapter"

ROUTER_SYSTEM = """
You are a routing controller for a technical-support system.
Return exactly one JSON object and no other text.
The JSON schema is:
{"route": "qa|support_specialist|tools|escalate", "reason": "short explanation"}

Routing policy:
- Use qa for questions answerable from trusted documentation.
- Use tools for live system, ticket, log, health, database, package, GPU, or network information.
- Use support_specialist for troubleshooting explanations and response synthesis.
- Use escalate for high-risk incidents, possible data loss, security breaches, or production corruption.
""".strip()

ROUTER_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "According to the deployment guide, which port should be exposed?",
    },
    {
        "role": "assistant",
        "content": '{"route": "qa", "reason": "The answer should come from trusted documentation."}',
    },
    {
        "role": "user",
        "content": "The health check reports that the database is degraded.",
    },
    {
        "role": "assistant",
        "content": '{"route": "tools", "reason": "Current system health must be inspected with tools."}',
    },
]

llm_router_tokenizer = None
llm_router_model = None
llm_router_device = None


def load_llm_router():
    global llm_router_tokenizer, llm_router_model, llm_router_device

    if llm_router_model is not None:
        return

    if torch.backends.mps.is_available():
        llm_router_device = torch.device("mps")
    elif torch.cuda.is_available():
        llm_router_device = torch.device("cuda")
    else:
        llm_router_device = torch.device("cpu")

    llm_router_tokenizer = AutoTokenizer.from_pretrained(
        LLM_ROUTER_ADAPTER_PATH
    )

    if llm_router_tokenizer.pad_token is None:
        llm_router_tokenizer.pad_token = llm_router_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_ROUTER_BASE_MODEL
    )

    llm_router_model = PeftModel.from_pretrained(
        base_model,
        LLM_ROUTER_ADAPTER_PATH,
    )

    llm_router_model.to(llm_router_device)
    llm_router_model.eval()


def build_llm_router_prompt(text: str):
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        *ROUTER_FEW_SHOT_EXAMPLES,
        {"role": "user", "content": text},
    ]


def validate_llm_router_response(raw_output: str):
    cleaned_output = raw_output.strip()

    if cleaned_output.startswith("```"):
        lines = cleaned_output.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_output = "\n".join(lines).strip()

    decision = json.loads(cleaned_output)

    if not isinstance(decision, dict):
        raise ValueError("The LLM router response must be a JSON object.")

    if set(decision) != {"route", "reason"}:
        raise ValueError("The JSON response must contain only route and reason.")

    if decision["route"] not in ALLOWED_ROUTES:
        raise ValueError(f"Unsupported route: {decision['route']}")

    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise ValueError("The routing reason must be a non-empty string.")

    return decision


def llm_router(text: str):
    load_llm_router()

    messages = build_llm_router_prompt(text)
    model_inputs = llm_router_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    model_inputs = {
        name: value.to(llm_router_device)
        for name, value in model_inputs.items()
    }

    prompt_length = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = llm_router_model.generate(
            **model_inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=llm_router_tokenizer.pad_token_id,
            eos_token_id=llm_router_tokenizer.eos_token_id,
        )

    raw_output = llm_router_tokenizer.decode(
        generated_ids[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()

    try:
        decision = validate_llm_router_response(raw_output)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return {
            "route": "support_specialist",
            "reason": "The LLM router returned invalid JSON, so the safe fallback was used.",
            "source": "llm_router_fallback",
            "valid_json": False,
            "error": str(error),
            "raw_output": raw_output,
        }

    decision.update(
        {
            "source": "llm_router",
            "valid_json": True,
        }
    )

    return decision


if __name__ == "__main__":
    test_messages = [
        "Production database corruption is suspected.",
        "According to the docs, which port is used?",
        "My PostgreSQL connection pool is full.",
        "CUDA reports an out-of-memory error.",
        "How can I reset my password?",
        "The application behaves strangely.",
    ]

    print("Router A: rules and classifier")
    for message in test_messages:
        result = baseline_router(message)
        print(f"Message: {message}\nRouting Decision: {result}\n")

    print("Router B: LoRA language model")
    for message in test_messages:
        result = llm_router(message)
        print(f"Message: {message}\nRouting Decision: {result}\n")
