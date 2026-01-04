# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

import pytest
from pathlib import Path
from sec_risk_api.risk_taxonomy import (
    RiskCategory,
    get_all_categories,
    get_category_description,
    get_category_keywords,
    validate_category,
    CATEGORY_METADATA
)
from sec_risk_api.prompt_manager import PromptManager


class TestRiskTaxonomy:
    """
    Tests for Subissue 3.1: Risk Taxonomy Schema.
    Verify that the taxonomy is well-defined, extensible, and type-safe.
    """

    def test_all_categories_are_defined(self) -> None:
        """
        Verify that all expected risk categories exist in the enum.
        """
        categories = get_all_categories()
        
        expected_categories = [
            "operational", "systematic", "geopolitical", "regulatory",
            "competitive", "technological", "human_capital", "financial",
            "reputational", "other"
        ]
        
        category_values = [c.value for c in categories]
        
        for expected in expected_categories:
            assert expected in category_values, f"Missing category: {expected}"

    def test_each_category_has_metadata(self) -> None:
        """
        Verify that every category has required metadata fields.
        """
        for category in RiskCategory:
            assert category in CATEGORY_METADATA, f"Missing metadata for {category}"
            
            metadata = CATEGORY_METADATA[category]
            assert "keywords" in metadata
            assert "severity_multiplier" in metadata
            assert "description" in metadata
            
            # Verify types
            assert isinstance(metadata["keywords"], list)
            assert isinstance(metadata["severity_multiplier"], (int, float))
            assert isinstance(metadata["description"], str)

    def test_get_category_description(self) -> None:
        """
        Verify that descriptions are retrievable and non-empty.
        """
        operational_desc = get_category_description(RiskCategory.OPERATIONAL)
        assert len(operational_desc) > 0
        # Description should be meaningful (at least 10 characters)
        assert len(operational_desc) >= 10

    def test_get_category_keywords(self) -> None:
        """
        Verify that keyword lists are meaningful and non-empty (except OTHER).
        """
        operational_keywords = get_category_keywords(RiskCategory.OPERATIONAL)
        assert isinstance(operational_keywords, list)
        assert len(operational_keywords) > 0
        assert "supply chain" in operational_keywords
        
        # OTHER category is expected to have empty keywords
        other_keywords = get_category_keywords(RiskCategory.OTHER)
        assert isinstance(other_keywords, list)

    def test_validate_category_with_valid_input(self) -> None:
        """
        Verify that valid category strings are correctly converted to enums.
        """
        result = validate_category("operational")
        assert result == RiskCategory.OPERATIONAL
        
        result = validate_category("GEOPOLITICAL")  # Case insensitive
        assert result == RiskCategory.GEOPOLITICAL

    def test_validate_category_with_invalid_input(self) -> None:
        """
        Verify that invalid category strings raise ValueError with helpful message.
        """
        with pytest.raises(ValueError) as exc_info:
            validate_category("INVALID_CATEGORY")
        
        error_message = str(exc_info.value)
        assert "Invalid category" in error_message
        assert "operational" in error_message  # Should list valid options

    def test_taxonomy_is_extensible(self) -> None:
        """
        Acceptance Test: Verify that adding a new category doesn't break core logic.
        
        This is a regression test. If we add a new category to the enum,
        this test should still pass (proving extensibility).
        """
        # Get all categories
        all_categories = get_all_categories()
        
        # Verify we can iterate and process each one
        for category in all_categories:
            # Core operations should work for all categories
            desc = get_category_description(category)
            keywords = get_category_keywords(category)
            
            assert isinstance(desc, str)
            assert isinstance(keywords, list)
            
            # Should be able to validate back to itself
            validated = validate_category(category.value)
            assert validated == category


class TestPromptManager:
    """
    Tests for Subissue 3.1: Prompt Versioning System.
    Verify that prompts can be loaded, versioned, and tracked.
    """

    def test_prompt_manager_initializes(self) -> None:
        """
        Verify that PromptManager can initialize and find prompts directory.
        """
        pm = PromptManager()
        assert pm.prompts_dir.exists()
        assert pm.prompts_dir.name == "prompts"

    def test_load_prompt_v1(self) -> None:
        """
        Verify that the v1 risk classification prompt can be loaded.
        """
        pm = PromptManager()
        prompt = pm.load_prompt("risk_classification", version=1)
        
        assert len(prompt) > 0
        assert "Risk Taxonomy" in prompt
        assert "OPERATIONAL" in prompt
        assert "JSON" in prompt  # Should specify JSON output format
        assert "evidence" in prompt  # Should require source citation

    def test_load_nonexistent_prompt_raises_error(self) -> None:
        """
        Verify that loading a non-existent prompt raises FileNotFoundError.
        """
        pm = PromptManager()
        
        with pytest.raises(FileNotFoundError):
            pm.load_prompt("nonexistent_prompt", version=1)

    def test_get_latest_version(self) -> None:
        """
        Verify that get_latest_version returns the highest version number.
        """
        pm = PromptManager()
        latest = pm.get_latest_version("risk_classification")
        
        # Should be at least version 1
        assert latest >= 1
        assert isinstance(latest, int)

    def test_load_latest(self) -> None:
        """
        Verify that load_latest loads the most recent prompt version.
        """
        pm = PromptManager()
        prompt = pm.load_latest("risk_classification")
        
        assert len(prompt) > 0
        assert "Risk Taxonomy" in prompt

    def test_list_available_prompts(self) -> None:
        """
        Verify that list_available_prompts returns expected prompt names.
        """
        pm = PromptManager()
        available = pm.list_available_prompts()
        
        assert isinstance(available, list)
        assert "risk_classification" in available

    def test_get_prompt_metadata(self) -> None:
        """
        Verify that metadata can be retrieved for a prompt version.
        """
        pm = PromptManager()
        metadata = pm.get_prompt_metadata("risk_classification", version=1)
        
        assert "prompt_name" in metadata
        assert "version" in metadata
        assert "file_path" in metadata
        assert "file_size_bytes" in metadata
        assert metadata["prompt_name"] == "risk_classification"
        assert metadata["version"] == 1
        assert metadata["file_size_bytes"] > 0

    def test_prompt_version_tracking(self, tmp_path: Path) -> None:
        """
        Test: Change prompt version — confirm changes are tracked.
        
        This test verifies that when a new prompt version is added,
        the system correctly identifies and loads it.
        """
        # Create a temporary prompts directory
        test_prompts_dir = tmp_path / "prompts"
        test_prompts_dir.mkdir()
        
        # Create v1
        v1_file = test_prompts_dir / "test_prompt_v1.txt"
        v1_file.write_text("This is version 1")
        
        # Create v2
        v2_file = test_prompts_dir / "test_prompt_v2.txt"
        v2_file.write_text("This is version 2 with improvements")
        
        # Initialize PromptManager with test directory
        pm = PromptManager(prompts_dir=test_prompts_dir)
        
        # Should detect v2 as latest
        latest_version = pm.get_latest_version("test_prompt")
        assert latest_version == 2
        
        # Should load v2 content
        latest_content = pm.load_latest("test_prompt")
        assert "version 2" in latest_content
        
        # Should still be able to load v1
        v1_content = pm.load_prompt("test_prompt", version=1)
        assert "version 1" in v1_content


class TestPromptRequirements:
    """
    Tests for Subissue 3.1 Success Conditions:
    - Prompt explicitly cites source chunks
    - Prompts stored in clear file structure
    - Prompt versions tracked in JOURNAL.md
    """

    def test_prompt_requires_source_citation(self) -> None:
        """
        Test: Output explicitly cites the responsible source chunk text.
        
        Verify that the prompt template requires the LLM to cite evidence.
        """
        pm = PromptManager()
        prompt = pm.load_latest("risk_classification")
        
        # Prompt should explicitly require evidence/citation
        assert "evidence" in prompt.lower() or "quote" in prompt.lower()
        assert "source" in prompt.lower() or "text" in prompt.lower()
        
        # Should specify JSON output with evidence field
        assert "json" in prompt.lower()

    def test_prompt_file_structure_is_clear(self) -> None:
        """
        Verify that prompts are stored in a clear, documented structure.
        """
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        # prompts/ directory should exist
        assert prompts_dir.exists()
        
        # Should have README
        readme = prompts_dir / "README.md"
        assert readme.exists()
        
        # Should have CHANGELOG
        changelog = prompts_dir / "CHANGELOG.md"
        assert changelog.exists()
        
        # Should have at least v1 of risk classification
        v1_prompt = prompts_dir / "risk_classification_v1.txt"
        assert v1_prompt.exists()

    def test_prompt_changelog_exists(self) -> None:
        """
        Verify that prompt changes are documented in CHANGELOG.md.
        """
        prompts_dir = Path(__file__).parent.parent / "prompts"
        changelog = prompts_dir / "CHANGELOG.md"
        
        assert changelog.exists()
        
        content = changelog.read_text()
        assert len(content) > 0
        assert "v1" in content.lower()
        assert "rationale" in content.lower()
