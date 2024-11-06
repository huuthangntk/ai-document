import asyncio
import shutil
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional

class BackupHandler:
    """Handler for documentation backups"""
    
    def __init__(
        self,
        base_dir: Path,
        project_name: str,
        logger: Optional[logging.Logger] = None
    ):
        self.base_dir = base_dir
        self.project_name = project_name
        self.logger = logger or logging.getLogger('docgen')
        
        self.backup_dir = self.base_dir / project_name / 'backup'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, docs_dir: Path) -> bool:
        """Create backup of existing documentation"""
        try:
            if not docs_dir.exists():
                return True
                
            # Generate backup name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"docs_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # Create backup using direct copy instead of run_in_executor
            shutil.copytree(docs_dir, backup_path, dirs_exist_ok=True)
            
            self.logger.info(f"Created backup at: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return False
    
    async def _async_copy(self, src: Path, dst: Path) -> None:
        """Asynchronously copy directory"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            shutil.copytree,
            src,
            dst,
            dirs_exist_ok=True
        )