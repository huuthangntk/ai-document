import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Custom theme for rich output
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "critical": "red reverse",
})

console = Console(theme=CUSTOM_THEME)

class DocumentationLogger:
    """Custom logger for documentation generation process"""
    
    def __init__(self, project_name: str, log_dir: Optional[Path] = None):
        self.project_name = project_name
        self.log_dir = log_dir or Path('/var/log/docgen')
        self.logger = logging.getLogger('docgen')
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self._configure_logging()
    
    def _configure_logging(self) -> None:
        """Configure logging handlers and formatters"""
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create log file path
        log_file = self.log_dir / f"{self.project_name}_docgen.log"
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Rich console handler
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """Return configured logger instance"""
        return self.logger

# Logging utility functions
def setup_logging(project_name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup and return logger instance"""
    doc_logger = DocumentationLogger(project_name, log_dir)
    return doc_logger.get_logger()

def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """Log exception with context"""
    logger.error(f"Error in {context}: {str(exc)}", exc_info=True)