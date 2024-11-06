import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import logging

class ErrorRecovery:
    """Handles error recovery and state persistence"""
    
    def __init__(self, project_dir: Path, logger: Optional[logging.Logger] = None):
        self.project_dir = project_dir
        self.logger = logger or logging.getLogger('docgen')
        self.state_file = project_dir / '.docgen_state'
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}
        
    async def save_state(self, failed_dir: str, error: str) -> None:
        """Save failed task state"""
        self.failed_tasks[failed_dir] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'retries': self.failed_tasks.get(failed_dir, {}).get('retries', 0) + 1
        }
        
        await self._write_state()
        
    async def _write_state(self) -> None:
        """Write state to file"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'failed_tasks': self.failed_tasks
            }
            
            async with asyncio.Lock():
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            
    async def recover_state(self) -> Dict[str, Dict[str, Any]]:
        """Recover previous state if exists"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.failed_tasks = state.get('failed_tasks', {})
                    return self.failed_tasks
        except Exception as e:
            self.logger.error(f"Failed to recover state: {e}")
            
        return {}
        
    async def clear_state(self) -> None:
        """Clear saved state"""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            self.failed_tasks.clear()
        except Exception as e:
            self.logger.error(f"Failed to clear state: {e}")