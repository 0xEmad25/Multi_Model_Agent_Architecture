import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

test_messages = [
    "Production database corruption is suspected.",
    "According to the docs, which port is used?",
    "My PostgreSQL connection pool is full.",
    "CUDA reports an out-of-memory error.",
    "How can I reset my password?",
    "The application behaves strangely.",
]

for message in test_messages:
    result = baseline_router(message)
    print(f"Message: {message}\nRouting Decision: {result}\n")