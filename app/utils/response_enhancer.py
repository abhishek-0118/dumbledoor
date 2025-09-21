"""Simplified response enhancement utilities"""

import logging
import re
from typing import Dict, Any

logger = logging.getLogger("app.utils.response_enhancer")


class ResponseEnhancer:
    """Simplified response enhancer focused on basic formatting only"""
    
    def __init__(self):
        self.basic_patterns = [
            # Function calls with parentheses
            (r'\b([a-zA-Z_][a-zA-Z0-9_]*)\(\)', r'`\1()`'),
            # File paths and extensions
            (r'\b([\w/.-]*\.(?:py|js|ts|go|java|cpp|h|jsx|tsx|rb|php|cs|rs|kt|swift|dart))\b', r'`\1`'),
            # Class names (simple PascalCase)
            (r'\b([A-Z][a-zA-Z0-9]*(?:Service|Controller|Manager|Handler|Component|Provider))\b', r'`\1`'),
        ]
        
    def enhance_response(self, response: str, context_summary: Dict[str, Any] = None) -> str:
        """Apply basic formatting enhancements only"""
        if not response or not response.strip():
            return response
        
        # Apply only basic code reference formatting
        enhanced = self._apply_basic_formatting(response)
        
        # Add simple context info if available  
        if context_summary and context_summary.get("total_documents", 0) > 0:
            enhanced = self._add_context_footer(enhanced, context_summary)
        
        return enhanced
    
    def enhance_streaming_chunk(self, chunk: str, session_id: str = "default") -> str:
        """For streaming, return chunk as-is to avoid interference"""
        return chunk
    
    def _apply_basic_formatting(self, text: str) -> str:
        """Apply only basic, safe formatting patterns"""
        # Don't process text inside code blocks
        if '```' in text:
            return text
        
        # Apply basic patterns
        for pattern, replacement in self.basic_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _add_context_footer(self, response: str, context_summary: Dict[str, Any]) -> str:
        """Add simple context information"""
        total_docs = context_summary.get("total_documents", 0)
        repos = context_summary.get("repositories", [])
        languages = context_summary.get("languages", [])
        
        if total_docs == 0:
            return response
            
        footer_parts = []
        footer_parts.append(f"**{total_docs} documents analyzed**")
        
        if repos:
            if len(repos) == 1:
                footer_parts.append(f"from **{repos[0]}**")
            else:
                footer_parts.append(f"from **{len(repos)} repositories**")
        
        if languages:
            if len(languages) == 1:
                footer_parts.append(f"({languages[0]})")
            elif len(languages) <= 3:
                footer_parts.append(f"({', '.join(languages)})")
            else:
                footer_parts.append(f"({len(languages)} languages)")
        
        footer = "\n\n---\n\n*Answer based on " + " ".join(footer_parts) + "*"
        return response + footer


# Global instance
response_enhancer = ResponseEnhancer()