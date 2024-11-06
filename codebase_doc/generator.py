# Location: ./codebase_doc/generator.py

import asyncio
import os
import shutil
from pathlib import Path
from typing import Set, List, Optional, Dict, Any, Tuple
import logging
from datetime import datetime

from .utils.backup_handler import BackupHandler
from .utils.ignore_parser import IgnorePatternHandler
from .utils.summary import SummaryGenerator

class DocumentationGenerator:
    """Asynchronous documentation generator for Python projects"""
    
    def __init__(
        self,
        project_name: str,
        output_dir: Path,
        force: bool = False,
        logger: Optional[logging.Logger] = None,
        max_concurrent: int = 20,
        timeout: int = 180
    ):
        """Initialize the documentation generator."""
        self.project_name = project_name
        self.base_output_dir = Path(output_dir)  # Ensure Path object
        self.force = force
        self.logger = logger or logging.getLogger('docgen')
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        # Initialize handlers
        self.backup_handler = BackupHandler(
            base_dir=self.base_output_dir,
            project_name=self.project_name,
            logger=self.logger
        )
        self.ignore_handler = IgnorePatternHandler()
        
        # Setup project directories
        self.project_dir = self.base_output_dir / self.project_name
        self.latest_dir = self.project_dir / 'latest'
        self.backup_dir = self.project_dir / 'backup'
        
        # Ensure directories exist
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Setup required directory structure"""
        try:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self.latest_dir.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to setup directories: {e}")
            raise
            
    async def generate(self, target_dirs: Set[Path]) -> bool:
        """Main generation method."""
        if not target_dirs:
            self.logger.error("No target directories provided")
            return False
            
        try:
            # Convert all paths to absolute
            target_dirs = {Path(d).resolve() for d in target_dirs}
            
            # Validate all target directories exist
            if not self._validate_targets(target_dirs):
                return False
                
            # Handle existing documentation
            if self.latest_dir.exists():
                if not self.force:
                    if not await self.backup_handler.create_backup(self.latest_dir):
                        self.logger.error("Failed to create backup")
                        return False
                        
                # Clean latest directory
                shutil.rmtree(self.latest_dir)
                self.latest_dir.mkdir(parents=True)
            
            # Process directories concurrently
            results = await self._process_directories(target_dirs)
            
            # Validate results
            if not results:
                self.logger.error("No results returned from processing")
                return False
                
            # Check for processing failures
            failed_dirs = [str(dir_path) for dir_path, success in results if not success]
            if failed_dirs:
                self.logger.error(f"Failed to process directories: {failed_dirs}")
                return False
                
            # Generate summary
            summary_generator = SummaryGenerator(self.latest_dir, self.logger)
            if not await summary_generator.generate():
                self.logger.error("Failed to generate summary")
                return False
                
            self.logger.info("Documentation generation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            return False
            
    def _validate_targets(self, target_dirs: Set[Path]) -> bool:
        """Validate target directories exist and are accessible."""
        try:
            for directory in target_dirs:
                if not directory.exists():
                    self.logger.error(f"Target directory does not exist: {directory}")
                    return False
                if not directory.is_dir():
                    self.logger.error(f"Target is not a directory: {directory}")
                    return False
                # Check read permissions
                if not os.access(directory, os.R_OK):
                    self.logger.error(f"No read permission for directory: {directory}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating targets: {e}")
            return False
        
    async def _process_directories(self, target_dirs: Set[Path]) -> List[Tuple[Path, bool]]:
        """Process directories concurrently with semaphore."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        
        for directory in target_dirs:
            task = asyncio.create_task(self._process_single_directory(directory, semaphore))
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_results = []
            for dir_path, result in zip(target_dirs, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing {dir_path}: {result}")
                    processed_results.append((dir_path, False))
                else:
                    processed_results.append((dir_path, result))
                    
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error during concurrent processing: {e}")
            return [(dir_path, False) for dir_path in target_dirs]
            
    async def _process_single_directory(
        self,
        directory: Path,
        semaphore: asyncio.Semaphore
    ) -> bool:
        """Process a single directory with timeout and retries."""
        async with semaphore:
            try:
                # Check for .docsignore
                ignore_file = directory / '.docsignore'
                if ignore_file.exists():
                    self.ignore_handler.load_patterns(ignore_file)
                
                # Generate documentation
                output_file = self._get_output_path(directory)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                success = await self._run_cdigest(
                    directory,
                    output_file,
                    ignore_patterns=self.ignore_handler.get_patterns()
                )
                
                if success:
                    self.logger.info(f"Generated documentation for {directory}")
                else:
                    self.logger.error(f"Failed to generate documentation for {directory}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Error processing directory {directory}: {e}")
                return False
                
    async def _run_cdigest(
        self,
        input_path: Path,
        output_file: Path,
        ignore_patterns: List[str],
        max_retries: int = 3
    ) -> bool:
        """Run cdigest command with retries.
        
        Args:
            input_path: Input directory path
            output_file: Output file path
            ignore_patterns: List of patterns to ignore
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if command was successful
        """
        for attempt in range(max_retries):
            try:
                cmd = [
                    'cdigest',
                    str(input_path),
                    '-o', 'markdown',
                    '-f', str(output_file),
                    '-d', '5',
                    '--max-size', '20480',
                    '--show-size'
                ]
                
                # Add ignore patterns
                for pattern in ignore_patterns:
                    cmd.extend(['--ignore', pattern])
                
                process = await asyncio.create_subprocess_exec(
                    'cdigest',
                    str(input_path),
                    '-o', 'markdown',
                    '-f', str(output_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    await asyncio.wait_for(process.wait(), timeout=self.timeout)
                    
                    return process.returncode == 0 or bool(output_file.exists())

                        
                except asyncio.TimeoutError:
                    if process.returncode is None:
                        process.terminate()
                        await process.wait()
                        
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return False
        
        return False
        
    def _get_output_path(self, input_path: Path) -> Path:
        """Generate output path for documentation file.
        
        Args:
            input_path: Input directory path
            
        Returns:
            Path: Output file path
        """
        relative_path = input_path.relative_to(input_path.parent)
        return self.latest_dir / f"{relative_path}.md"