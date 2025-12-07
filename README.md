# PiiSanitizer

A Python utility for identifying, anonymizing, and restoring Personally Identifiable Information (PII) in text. It supports multiple detection backends including Microsoft Presidio (default & transformer-based) and AWS Bedrock Guardrails.

This tool is particularly useful for LLM workflows where you need to sanitize sensitive data before sending it to a model provider, and restore it in the response.

## Features

- **Multiple Backends**:
  - **Presidio (Default)**: Fast, local PII detection using Spacy and Presidio.
  - **Presidio Transformer**: Higher accuracy using HuggingFace transformer models (specifically `StanfordAIMI/stanford-deidentifier-base`).
  - **AWS Bedrock Guardrails**: Leverages AWS managed guardrails for PII detection.
- **Reversible Anonymization**: Maintains a mapping of original values to placeholders (e.g., `{PERSON_0}`) to restore data later.
- **Decorator Support**: Easily wrap functions to automatically sanitize input and restore output.

## Installation

1. Install the required Python packages:

```bash
pip install -r requirement.txt
```

2. Download the Spacy language model (required for Presidio modes):

```bash
python -m spacy download en_core_web_sm
```

## Usage

### 1. Basic Usage (Default Presidio)

```python
from pii_sanitizer import PiiSanitizer, RunningMode

# Initialize
sanitizer = PiiSanitizer(running_mode=RunningMode.PRESIDIO)

# Anonymize
original_text = "My name is John Doe and my email is john@example.com."
anonymized_text, mapping = sanitizer.anonymize_message(original_text)

print(f"Anonymized: {anonymized_text}")
# Output: My name is {PERSON_0} and my email is {EMAIL_ADDRESS_0}.

# Restore (Deanonymize)
restored_text = sanitizer.deanonymize_message(anonymized_text, mapping)
print(f"Restored: {restored_text}")
# Output: My name is John Doe and my email is john@example.com.
```

### 2. Using Transformer Model (Higher Accuracy)

This mode uses the `StanfordAIMI/stanford-deidentifier-base` model via Presidio for better entity recognition in medical or complex contexts.

```python
sanitizer = PiiSanitizer(running_mode=RunningMode.PRESIDIO_TRANSFORMER)

text = "Patient Roy visited Dr. Smith."
anonymized, mapping = sanitizer.anonymize_message(text)
```

### 3. Using AWS Bedrock Guardrails

Requires AWS credentials and a configured Bedrock Guardrail.

```python
sanitizer = PiiSanitizer(
    running_mode=RunningMode.BEDROCK_GUARDRAIL,
    region="us-east-1",
    bedrock_guardrail_arn="arn:aws:bedrock:us-east-1:123456789012:guardrail/YOUR_GUARDRAIL_ID",
    bedrock_guardrail_arn_version="1",
    profile="default"  # Optional: AWS CLI profile name
    # Or provide specific credentials:
    # access_key="...",
    # secret_key="...",
    # session_token="..."
)

text = "Call me at 555-0123"
anonymized, mapping = sanitizer.anonymize_message(text)
```

### 4. Decorator for Automatic Sanitization

You can use the `@wrap_rephrase` decorator to automatically sanitize arguments passed to a function and restore PII in the return value. This is ideal for wrapping LLM calls.

```python
sanitizer = PiiSanitizer(running_mode=RunningMode.PRESIDIO)

@sanitizer.wrap_rephrase()
def chat_with_llm(message: str) -> str:
    # The 'message' here is already anonymized (e.g., "Hello {PERSON_0}")
    print(f"Processing: {message}")
    
    # Simulate LLM logic (e.g., returning the text or a modification)
    return f"I received your message: {message}"

# Usage
response = chat_with_llm("Hello, I am Alice.")

# The decorator handles the flow:
# 1. "Hello, I am Alice." -> "Hello, I am {PERSON_0}."
# 2. chat_with_llm runs with anonymized text.
# 3. Returns "I received your message: Hello, I am {PERSON_0}."
# 4. Decorator restores PII -> "I received your message: Hello, I am Alice."

print(response)
```

## Configuration

The `PiiSanitizer` class constructor accepts the following arguments:

- `running_mode`: Enum `RunningMode` (PRESIDIO, PRESIDIO_TRANSFORMER, BEDROCK_GUARDRAIL).
- `region` (Bedrock only): AWS region.
- `bedrock_guardrail_arn` (Bedrock only): ARN of the guardrail.
- `bedrock_guardrail_arn_version` (Bedrock only): Version of the guardrail.
- `profile` (Optional): AWS profile name.
- `access_key`, `secret_key`, `session_token` (Optional): AWS credentials.
