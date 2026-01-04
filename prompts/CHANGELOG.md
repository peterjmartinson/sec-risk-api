# Prompt Version History

## v1.0 (2026-01-03)

**Status**: Initial Release

**Rationale**: 
First iteration of the risk classification prompt. Designed to provide clear category definitions with concrete examples, enforce JSON output format, and require source citation for every classification.

**Key Features**:
- 10 mutually exclusive risk categories
- Confidence scoring (0.0-1.0)
- Mandatory evidence citation
- Explicit handling of multi-category and ambiguous cases
- JSON output schema for easy parsing

**Testing Notes**:
- Designed for use with GPT-4 or Claude-3 class models
- Expected accuracy: >85% on hand-labeled samples
- Typical confidence scores: 0.75-0.95 for clear cases, 0.4-0.7 for ambiguous

**Known Limitations**:
- Does not yet handle multi-label classification (intentional for v1)
- Edge case: Risks that are equally operational + geopolitical may be inconsistent
- No explicit handling of risk severity scoring (deferred to Subissue 3.2)

**Future Improvements**:
- Add few-shot examples for edge cases
- Consider chain-of-thought reasoning for complex risks
- Explore multi-label classification for v2 if needed
