# Risk Classification Prompts

This directory contains versioned system prompts for LLM-based risk classification.

## Structure

- `risk_classification_v1.txt` - Initial prompt for risk taxonomy classification
- `risk_classification_v2.txt` - Refined prompt (if needed)
- `CHANGELOG.md` - Version history and rationale for prompt changes

## Design Principles

1. **Explicit Instructions**: Prompts clearly define each risk category with examples
2. **Source Citation**: LLM must quote the specific text that justifies the classification
3. **Format Specification**: Output must be parseable JSON with strict schema
4. **Ambiguity Handling**: Instructions for multi-category risks or unclear cases
5. **Extensibility**: New categories can be added without rewriting core logic

## Usage

Prompts are loaded by `src/sec_risk_api/prompt_manager.py` and applied via the classification pipeline.
