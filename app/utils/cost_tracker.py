"""Cost tracking and logging utilities"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import tiktoken
from ..constants import MODEL_COSTS, DEFAULT_MODEL_COSTS, EMBEDDING_COSTS, DEFAULT_EMBEDDING_COSTS


@dataclass
class CostEntry:
    """Single cost tracking entry"""
    timestamp: str
    session_id: str
    operation_type: str  # "chat", "embedding", "search"
    provider: str
    model: str
    input_tokens: int
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    query: Optional[str] = None
    context_size: int = 0


class CostTracker:
    """Enhanced cost tracking with persistent logging"""
    
    def __init__(self, cost_log_file: str = "./logs/cost_history.log", 
                 session_id: Optional[str] = None):
        self.cost_log_file = Path(cost_log_file)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger("app.cost_tracker")
        
        # Ensure log directory exists
        self.cost_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Session totals
        self.session_costs = {
            "chat": 0.0,
            "embedding": 0.0,
            "total": 0.0,
            "token_count": 0
        }
    
    def track_chat_cost(self, provider: str, model: str, input_text: str, 
                       output_text: str = "", query: str = None, 
                       context_size: int = 0) -> Dict[str, Any]:
        """Track chat LLM costs with detailed logging"""
        
        input_tokens = self._count_tokens(input_text, provider, model)
        output_tokens = self._count_tokens(output_text, provider, model) if output_text else 0
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost_per_1k = self._get_chat_cost_per_1k(model, provider)
        estimated_cost = (total_tokens / 1000) * cost_per_1k
        
        # Create cost entry
        entry = CostEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            operation_type="chat",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            query=query[:100] if query else None,  # Truncate for privacy
            context_size=context_size
        )
        
        # Update session totals
        self.session_costs["chat"] += estimated_cost
        self.session_costs["total"] += estimated_cost
        self.session_costs["token_count"] += total_tokens
        
        # Log the cost entry
        self._log_cost_entry(entry)
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "cost_per_1k_tokens": cost_per_1k,
            "session_total": self.session_costs["total"]
        }
    
    def track_embedding_cost(self, provider: str, model: str, texts: list, 
                           operation: str = "embedding") -> Dict[str, Any]:
        """Track embedding costs"""
        
        total_text = " ".join(texts) if isinstance(texts, list) else texts
        input_tokens = self._count_tokens(total_text, provider, model)
        
        # Calculate cost
        cost_per_1k = self._get_embedding_cost_per_1k(model, provider)
        estimated_cost = (input_tokens / 1000) * cost_per_1k
        
        # Create cost entry
        entry = CostEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            operation_type="embedding",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            total_tokens=input_tokens,
            estimated_cost=estimated_cost,
            context_size=len(texts) if isinstance(texts, list) else 1
        )
        
        # Update session totals
        self.session_costs["embedding"] += estimated_cost
        self.session_costs["total"] += estimated_cost
        self.session_costs["token_count"] += input_tokens
        
        # Log the cost entry
        self._log_cost_entry(entry)
        
        return {
            "input_tokens": input_tokens,
            "estimated_cost": estimated_cost,
            "cost_per_1k_tokens": cost_per_1k,
            "session_total": self.session_costs["total"]
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of costs for current session"""
        return {
            "session_id": self.session_id,
            "session_costs": self.session_costs.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily cost summary from log file"""
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        
        daily_costs = {
            "chat": 0.0,
            "embedding": 0.0,
            "total": 0.0,
            "token_count": 0,
            "request_count": 0
        }
        
        if not self.cost_log_file.exists():
            return daily_costs
        
        try:
            with self.cost_log_file.open('r') as f:
                for line in f:
                    try:
                        entry_data = json.loads(line.strip())
                        entry_date = entry_data["timestamp"][:10]  # Extract date part
                        
                        if entry_date == target_date:
                            daily_costs[entry_data["operation_type"]] += entry_data["estimated_cost"]
                            daily_costs["total"] += entry_data["estimated_cost"]
                            daily_costs["token_count"] += entry_data["total_tokens"]
                            daily_costs["request_count"] += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            self.logger.warning(f"Failed to read cost log: {e}")
        
        return daily_costs
    
    def _count_tokens(self, text: str, provider: str, model: str) -> int:
        """Count tokens using appropriate tokenizer"""
        if not text:
            return 0
            
        try:
            if provider.lower() == "openai":
                if "gpt-4" in model.lower():
                    encoding = tiktoken.encoding_for_model("gpt-4")
                elif "gpt-3.5" in model.lower():
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:
                # Use cl100k_base for other providers as approximation
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}")
            # Fallback to word count approximation
            return int(len(text.split()) * 1.3)
    
    def _get_chat_cost_per_1k(self, model: str, provider: str) -> float:
        """Get chat model cost per 1K tokens"""
        provider_key = provider.lower()
        if provider_key in MODEL_COSTS:
            costs = MODEL_COSTS[provider_key]
            for model_key, cost in costs.items():
                if model_key in model.lower():
                    return cost
            return DEFAULT_MODEL_COSTS.get(provider_key, DEFAULT_MODEL_COSTS["default"])
        return DEFAULT_MODEL_COSTS["default"]
    
    def _get_embedding_cost_per_1k(self, model: str, provider: str) -> float:
        """Get embedding model cost per 1K tokens"""
        provider_key = provider.lower()
        if provider_key == "gemini":
            provider_key = "google"
        
        if provider_key in EMBEDDING_COSTS:
            costs = EMBEDDING_COSTS[provider_key]
            for model_key, cost in costs.items():
                if model_key in model.lower():
                    return cost
            return DEFAULT_EMBEDDING_COSTS.get(provider_key, 0.0)
        return DEFAULT_EMBEDDING_COSTS.get(provider_key, 0.0)
    
    def _log_cost_entry(self, entry: CostEntry):
        """Log cost entry to file and logger"""
        # Log to file as JSON
        try:
            with self.cost_log_file.open('a') as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write cost entry to file: {e}")
        
        # Log to standard logger
        self.logger.info(
            f"Cost tracking - {entry.operation_type}: "
            f"{entry.provider}/{entry.model} - "
            f"{entry.total_tokens} tokens - "
            f"${entry.estimated_cost:.6f} - "
            f"Session total: ${self.session_costs['total']:.6f}"
        )


def setup_cost_tracking(enable_tracking: bool = True, 
                       cost_log_file: str = "./logs/cost_history.log") -> Optional[CostTracker]:
    """Setup global cost tracking"""
    if not enable_tracking:
        return None
    
    return CostTracker(cost_log_file=cost_log_file)
