from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

class SummaryGenerator:
    """Generates documentation summary with tree structure"""
    
    def __init__(self, docs_dir: Path, logger: logging.Logger):
        self.docs_dir = docs_dir
        self.logger = logger
    
    async def generate(self) -> bool:
        """Generate documentation summary"""
        try:
            markdown_files = sorted([
                f for f in self.docs_dir.glob('*.md')
                if f.name != 'readme.md'
            ])
            
            if not markdown_files:
                self.logger.warning("No documentation files found to summarize")
                return False
            
            tree = self._build_tree(markdown_files)
            content = self._generate_content(tree, markdown_files)
            
            await self._write_summary(content)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return False
    
    def _build_tree(self, files: List[Path]) -> Dict[str, Any]:
        """Build documentation tree structure"""
        tree = {}
        for doc_file in files:
            parts = doc_file.stem.split('.')
            current = tree
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = doc_file
        
        return tree
    
    def _generate_content(
        self,
        tree: Dict[str, Any],
        files: List[Path]
    ) -> List[str]:
        """Generate summary content"""
        content = [
            "# Project Documentation\n\n",
            "## Documentation Files\n\n"
        ]
        
        def write_tree(tree: Dict[str, Any], level: int = 0) -> List[str]:
            result = []
            indent = "  " * level
            
            for key, value in sorted(tree.items()):
                if isinstance(value, dict):
                    result.append(f"{indent}- **{key}**\n")
                    result.extend(write_tree(value, level + 1))
                else:
                    description = self._get_file_description(value)
                    result.append(
                        f"{indent}- [{key}]({value.name}): {description}\n"
                    )
            return result
        
        content.extend(write_tree(tree))
        content.extend([
            "\n## Statistics\n\n",
            f"- Total documentation files: {len(files)}\n",
            f"- Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ])
        
        return content
    
    def _get_file_description(self, file_path: Path) -> str:
        """Get first meaningful line from file as description"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        return (
                            f"{line[:100]}..."
                            if len(line) > 100
                            else line
                        )
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return "No description available"
        return "Empty file"
    
    async def _write_summary(self, content: List[str]) -> None:
        """Write summary to readme.md"""
        readme_path = self.docs_dir / 'readme.md'
        try:
            with open(readme_path, 'w') as f:
                f.write(''.join(content))
            self.logger.info("Documentation summary generated successfully")
        except Exception as e:
            self.logger.error(f"Failed to write summary: {e}")
            raise