"""Tools registry for the auction simulator."""

from .builtin.bid import bid_tool
from .builtin.call import call_tool
from .builtin.fold import fold_tool
from .builtin.status import status_tool
from .builtin.check_termination import check_termination_tool

# Registry of local tools
LOCAL_TOOLS = {
    "BID": bid_tool,
    "CALL": call_tool,
    "FOLD": fold_tool,
    "STATUS": status_tool,
    "CHECK_TERMINATION": check_termination_tool,
}

# Tools that use MCP
MCP_TOOLS = {
    "ASK_SELLER",
}

def is_local_tool(tool_name: str) -> bool:
    """Check if a tool is implemented locally."""
    return tool_name in LOCAL_TOOLS

def is_mcp_tool(tool_name: str) -> bool:
    """Check if a tool uses MCP."""
    return tool_name in MCP_TOOLS

__all__ = ["LOCAL_TOOLS", "MCP_TOOLS", "is_local_tool", "is_mcp_tool"] 