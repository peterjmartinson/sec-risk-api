# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Risk Taxonomy Schema for SEC Filing Classification.

This module defines the proprietary risk categories used to classify
Item 1A: Risk Factors from SEC filings.

Design Principles:
- Mutually Exclusive Categories: Each risk should fit primarily into one category
- Extensible: New categories can be added without breaking existing logic
- Hierarchical: Categories can have subcategories for future refinement
"""

from typing import List, Dict, Any
from enum import Enum


class RiskCategory(str, Enum):
    """
    Proprietary risk taxonomy for SEC filings.
    
    These categories are designed to capture the primary risk vectors
    that financial institutions care about when modeling company exposure.
    """
    
    # Core operational execution risks
    OPERATIONAL = "operational"
    """
    Risks related to internal business operations, processes, and execution.
    Examples: Supply chain disruptions, manufacturing delays, quality control,
    workforce management, IT infrastructure failures.
    """
    
    # Broad market and economic forces
    SYSTEMATIC = "systematic"
    """
    Risks from macroeconomic conditions and market-wide forces beyond
    company control.
    Examples: Recession, inflation, interest rates, market volatility,
    credit market conditions, economic downturns.
    """
    
    # International relations and political instability
    GEOPOLITICAL = "geopolitical"
    """
    Risks from international conflicts, trade policy, and political instability.
    Examples: War, trade disputes, sanctions, tariffs, international tensions,
    regional conflicts, foreign policy changes.
    """
    
    # Government rules and compliance
    REGULATORY = "regulatory"
    """
    Risks from government regulation, compliance requirements, and legal frameworks.
    Examples: New laws, regulatory changes, compliance costs, industry-specific
    regulations, environmental regulations, data privacy laws.
    """
    
    # Market rivalry and competitive positioning
    COMPETITIVE = "competitive"
    """
    Risks from competitors, market share erosion, and competitive dynamics.
    Examples: New entrants, price competition, product substitution,
    loss of competitive advantage, market saturation.
    """
    
    # Innovation and disruption threats
    TECHNOLOGICAL = "technological"
    """
    Risks from technological change, innovation, and digital disruption.
    Examples: Obsolescence, rapid tech evolution, cybersecurity threats,
    failure to innovate, emerging technologies disrupting business model.
    """
    
    # Human capital and organizational culture
    HUMAN_CAPITAL = "human_capital"
    """
    Risks related to workforce, talent, and organizational capabilities.
    Examples: Key employee retention, talent acquisition, labor disputes,
    workplace culture, succession planning, skill gaps.
    """
    
    # Financial structure and capital management
    FINANCIAL = "financial"
    """
    Risks related to company financial structure, debt, and capital management.
    Examples: Liquidity constraints, debt covenants, capital requirements,
    foreign exchange exposure, commodity price exposure.
    """
    
    # Brand, reputation, and stakeholder perception
    REPUTATIONAL = "reputational"
    """
    Risks to company reputation, brand value, and stakeholder trust.
    Examples: PR crises, social media backlash, ESG controversies,
    product recalls affecting brand, customer trust erosion.
    """
    
    # Catch-all for risks that don't fit primary categories
    OTHER = "other"
    """
    Risks that don't clearly fit into primary categories or are highly
    company-specific.
    """


# Metadata for each category to support extensibility and documentation
CATEGORY_METADATA: Dict[RiskCategory, Dict[str, Any]] = {
    RiskCategory.OPERATIONAL: {
        "keywords": ["supply chain", "manufacturing", "production", "operations", 
                    "logistics", "quality", "facilities", "infrastructure"],
        "severity_multiplier": 1.2,  # Higher operational risks often more immediate
        "description": "Internal execution and business process risks"
    },
    RiskCategory.SYSTEMATIC: {
        "keywords": ["economic", "recession", "inflation", "interest rate", 
                    "market conditions", "macroeconomic", "downturn"],
        "severity_multiplier": 1.0,
        "description": "Broad economic and market forces"
    },
    RiskCategory.GEOPOLITICAL: {
        "keywords": ["war", "conflict", "international", "geopolitical", 
                    "trade war", "sanctions", "tariff", "foreign policy"],
        "severity_multiplier": 0.9,  # Often harder to predict/quantify
        "description": "International relations and political instability"
    },
    RiskCategory.REGULATORY: {
        "keywords": ["regulation", "regulatory", "compliance", "law", 
                    "government", "legal", "policy", "legislation"],
        "severity_multiplier": 1.1,
        "description": "Government regulation and compliance requirements"
    },
    RiskCategory.COMPETITIVE: {
        "keywords": ["competition", "competitor", "market share", "competitive", 
                    "rival", "pricing pressure", "new entrant"],
        "severity_multiplier": 1.0,
        "description": "Market competition and competitive positioning"
    },
    RiskCategory.TECHNOLOGICAL: {
        "keywords": ["technology", "innovation", "obsolescence", "digital", 
                    "cybersecurity", "disruption", "automation"],
        "severity_multiplier": 1.3,  # Tech disruption can be severe
        "description": "Technological change and digital disruption"
    },
    RiskCategory.HUMAN_CAPITAL: {
        "keywords": ["employee", "workforce", "talent", "personnel", 
                    "labor", "retention", "hiring", "human capital"],
        "severity_multiplier": 0.8,
        "description": "Workforce and organizational capability risks"
    },
    RiskCategory.FINANCIAL: {
        "keywords": ["debt", "liquidity", "capital", "financial", 
                    "credit", "funding", "covenant", "cash flow"],
        "severity_multiplier": 1.4,  # Financial risks often existential
        "description": "Financial structure and capital management"
    },
    RiskCategory.REPUTATIONAL: {
        "keywords": ["reputation", "brand", "trust", "image", 
                    "perception", "esg", "social", "ethical"],
        "severity_multiplier": 0.9,
        "description": "Brand and stakeholder perception risks"
    },
    RiskCategory.OTHER: {
        "keywords": [],
        "severity_multiplier": 1.0,
        "description": "Miscellaneous or company-specific risks"
    }
}


def get_all_categories() -> List[RiskCategory]:
    """Returns all defined risk categories."""
    return list(RiskCategory)


def get_category_description(category: RiskCategory) -> str:
    """
    Get the detailed description for a risk category.
    
    Args:
        category: The risk category
    
    Returns:
        Human-readable description of the category
    """
    return CATEGORY_METADATA[category]["description"]


def get_category_keywords(category: RiskCategory) -> List[str]:
    """
    Get keyword hints for a risk category.
    
    These keywords can be used for:
    - Prompt engineering (giving LLM examples)
    - Validation (checking if classification makes sense)
    - Debugging (understanding why a category was chosen)
    
    Args:
        category: The risk category
    
    Returns:
        List of keyword strings associated with this category
    """
    return CATEGORY_METADATA[category]["keywords"]


def validate_category(category_str: str) -> RiskCategory:
    """
    Validate and convert a string to a RiskCategory.
    
    Args:
        category_str: String representation of category
    
    Returns:
        Valid RiskCategory enum
    
    Raises:
        ValueError: If category_str is not a valid category
    """
    try:
        return RiskCategory(category_str.lower())
    except ValueError:
        valid_categories = [c.value for c in RiskCategory]
        raise ValueError(
            f"Invalid category '{category_str}'. "
            f"Must be one of: {', '.join(valid_categories)}"
        )
