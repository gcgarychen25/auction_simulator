"""Configuration utilities for the auction simulator."""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_tool_backend(tool_name: str, config: Dict[str, Any]) -> str:
    """
    Get the backend type for a tool (local or mcp).
    
    Args:
        tool_name: Name of the tool
        config: Configuration dictionary
        
    Returns:
        Backend type ("local" or "mcp")
    """
    tools_config = config.get("tools", {})
    return tools_config.get(tool_name, "local") 