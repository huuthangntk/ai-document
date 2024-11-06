from pathlib import Path
from typing import List, Set
import fnmatch
import os

class IgnorePatternHandler:
    """Handler for .docsignore patterns"""
    
    DEFAULT_PATTERNS = {
        'venv', '.venv', '.git', '__pycache__',
        'node_modules', '.idea', '.vscode',
        'build', 'dist', 'egg-info',
        '*.pyc'  # Added common file pattern
    }
    
    def __init__(self):
        self.patterns: Set[str] = self.DEFAULT_PATTERNS.copy()
    
    def load_patterns(self, ignore_file: Path) -> None:
        """Load patterns from .docsignore file"""
        try:
            with open(ignore_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.patterns.add(line)
        except Exception as e:
            # Fall back to default patterns on error
            self.patterns = self.DEFAULT_PATTERNS.copy()
    
    def get_patterns(self) -> List[str]:
        """Get current ignore patterns"""
        return list(self.patterns)
    
    def should_ignore(self, path: str) -> bool:
        """Check if path should be ignored"""
        path = str(path)
        path_parts = path.split(os.sep)
        
        for pattern in self.patterns:
            # Check if pattern matches full path
            if fnmatch.fnmatch(path, pattern):
                return True
            
            # Check if pattern matches any path component
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
                
        return False
    
    def reset_to_defaults(self) -> None:
        """Reset patterns to defaults"""
        self.patterns = self.DEFAULT_PATTERNS.copy()