# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Prompt Manager for version-controlled risk classification prompts.

This module handles loading, versioning, and tracking of system prompts
used for LLM-based risk classification.
"""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages versioned prompts for risk classification.
    
    Design Principles:
    - Prompts are stored as text files in the prompts/ directory
    - Version tracking enables A/B testing and prompt evolution
    - All prompt changes must be documented with rationale
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files.
                        Defaults to ./prompts/ relative to project root.
        """
        if prompts_dir is None:
            # Default to prompts/ directory at project root
            self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        if not self.prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found: {self.prompts_dir}"
            )
        
        logger.info(f"PromptManager initialized: {self.prompts_dir}")
    
    def load_prompt(self, prompt_name: str, version: int = 1) -> str:
        """
        Load a specific prompt version.
        
        Args:
            prompt_name: Base name of the prompt (e.g., "risk_classification")
            version: Version number (default: 1)
        
        Returns:
            The prompt text as a string
        
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        prompt_file = self.prompts_dir / f"{prompt_name}_v{version}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_file}\n"
                f"Available prompts: {self.list_available_prompts()}"
            )
        
        logger.info(f"Loading prompt: {prompt_file.name}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        return prompt_text
    
    def get_latest_version(self, prompt_name: str) -> int:
        """
        Get the latest version number for a prompt.
        
        Args:
            prompt_name: Base name of the prompt
        
        Returns:
            The highest version number found
        
        Raises:
            FileNotFoundError: If no versions of the prompt exist
        """
        # Find all files matching the pattern
        pattern = f"{prompt_name}_v*.txt"
        matching_files = list(self.prompts_dir.glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(
                f"No versions found for prompt: {prompt_name}"
            )
        
        # Extract version numbers
        versions = []
        for file in matching_files:
            # Extract version from filename like "risk_classification_v2.txt"
            try:
                version_str = file.stem.split('_v')[-1]
                versions.append(int(version_str))
            except (ValueError, IndexError):
                logger.warning(f"Couldn't parse version from: {file.name}")
        
        if not versions:
            raise FileNotFoundError(
                f"No valid version numbers found for: {prompt_name}"
            )
        
        latest = max(versions)
        logger.info(f"Latest version of '{prompt_name}': v{latest}")
        return latest
    
    def load_latest(self, prompt_name: str) -> str:
        """
        Load the latest version of a prompt.
        
        Args:
            prompt_name: Base name of the prompt
        
        Returns:
            The prompt text from the latest version
        """
        latest_version = self.get_latest_version(prompt_name)
        return self.load_prompt(prompt_name, version=latest_version)
    
    def list_available_prompts(self) -> list[str]:
        """
        List all available prompt base names.
        
        Returns:
            List of unique prompt base names (without versions)
        """
        prompt_files = self.prompts_dir.glob("*_v*.txt")
        base_names = set()
        
        for file in prompt_files:
            # Extract base name from "risk_classification_v1.txt" -> "risk_classification"
            base_name = '_'.join(file.stem.split('_v')[:-1])
            base_names.add(base_name)
        
        return sorted(list(base_names))
    
    def get_prompt_metadata(self, prompt_name: str, version: int) -> dict[str, any]:
        """
        Get metadata about a specific prompt version.
        
        Args:
            prompt_name: Base name of the prompt
            version: Version number
        
        Returns:
            Dictionary with metadata (file_path, size, etc.)
        """
        prompt_file = self.prompts_dir / f"{prompt_name}_v{version}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        return {
            "prompt_name": prompt_name,
            "version": version,
            "file_path": str(prompt_file),
            "file_size_bytes": prompt_file.stat().st_size,
            "exists": True
        }
