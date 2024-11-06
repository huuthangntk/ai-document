# Codebase Analysis for: .

## Directory Structure

```
└── .
    ├── __init__.py (0 bytes)
    ├── generator.py (10472 bytes)
    ├── tests
    │   ├── test_cli.py (3497 bytes)
    │   ├── test_ignore.py (1840 bytes)
    │   └── test_generator.py (2853 bytes)
    ├── cli.py (4382 bytes)
    └── utils
        ├── processor.py (2208 bytes)
        ├── backup_handler.py (1742 bytes)
        ├── __init__.py (0 bytes)
        ├── recovery.py (2222 bytes)
        ├── ignore_parser.py (1766 bytes)
        ├── progress.py (2834 bytes)
        ├── logger.py (2413 bytes)
        └── summary.py (3891 bytes)
```

## Summary

- Total files: 14
- Total directories: 2
- Analyzed size: 39.18 KB
- Total text file size (including ignored): 14.51 KB
- Total tokens: 7820
- Analyzed text content size: 39.17 KB

## File Contents

### ./__init__.py

```

```

### ./generator.py

```
import asyncio
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
        """Initialize the documentation generator.
        
        Args:
            project_name: Name of the project
            output_dir: Base directory for output
            force: Whether to force overwrite existing docs
            logger: Optional logger instance
            max_concurrent: Maximum concurrent tasks
            timeout: Timeout in seconds for each operation
        """
        self.project_name = project_name
        self.base_output_dir = output_dir
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
        """Main generation method.
        
        Args:
            target_dirs: Set of target directories to process
            
        Returns:
            bool: True if generation was successful, False otherwise
        """
        if not target_dirs:
            self.logger.error("No target directories provided")
            return False
            
        try:
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
            failed_dirs = [dir_path for dir_path, success in results if not success]
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
        for directory in target_dirs:
            if not directory.exists():
                self.logger.error(f"Target directory does not exist: {directory}")
                return False
            if not directory.is_dir():
                self.logger.error(f"Target is not a directory: {directory}")
                return False
        return True
        
    async def _process_directories(self, target_dirs: Set[Path]) -> List[Tuple[Path, bool]]:
        """Process directories concurrently with semaphore.
        
        Args:
            target_dirs: Set of directories to process
            
        Returns:
            List of tuples containing (directory_path, success_status)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        
        for directory in target_dirs:
            task = self._process_single_directory(directory, semaphore)
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
        """Process a single directory with timeout and retries.
        
        Args:
            directory: Directory to process
            semaphore: Semaphore for concurrency control
            
        Returns:
            bool: True if processing was successful
        """
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
```

### ./tests/test_cli.py

```
# Location: ./codebase_doc/tests/test_cli.py

import asyncio
import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from codebase_doc.cli import main

@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner"""
    return CliRunner()

@pytest.fixture
def temp_targets(tmp_path: Path) -> Path:
    """Create temporary target directories and files"""
    # Create test directories
    (tmp_path / "dir1" / "src").mkdir(parents=True)
    (tmp_path / "dir2" / "src").mkdir(parents=True)
    
    # Create target file
    target_file = tmp_path / "targets.txt"
    target_file.write_text(f"{tmp_path}/dir1/src\n{tmp_path}/dir2/src")
    
    return tmp_path

@pytest.fixture
def mock_event_loop():
    """Mock event loop for testing"""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.run_until_complete.return_value = True
    loop.close = MagicMock()
    return loop

def test_cli_help(runner: CliRunner) -> None:
    """Test help output"""
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert '--target' in result.output
    assert '--file' in result.output
    assert '--force' in result.output
    assert '--output-dir' in result.output

def test_cli_target_option(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test target directory option"""
    with patch('codebase_doc.generator.DocumentationGenerator') as mock_gen:
        # Setup mock generator
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=True)
        
        # Mock event loop creation
        with patch('asyncio.new_event_loop', return_value=mock_event_loop):
            dir1 = temp_targets / "dir1" / "src"
            dir1.mkdir(parents=True, exist_ok=True)
            result = runner.invoke(main, ['--target', str(dir1)])
            
            assert result.exit_code == 0
            assert mock_gen.called
            mock_gen.assert_called_once()

def test_cli_file_option(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test target file option"""
    with patch('codebase_doc.generator.DocumentationGenerator') as mock_gen:
        # Setup mock generator
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=True)
        
        # Mock event loop creation
        with patch('asyncio.new_event_loop', return_value=mock_event_loop):
            target_file = temp_targets / "targets.txt"
            result = runner.invoke(main, ['--file', str(target_file)])
            
            assert result.exit_code == 0
            assert mock_gen.called
            mock_gen.assert_called_once()

def test_cli_force_flag(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test force flag functionality"""
    with patch('codebase_doc.generator.DocumentationGenerator') as mock_gen:
        # Setup mock generator
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=True)
        
        # Mock event loop creation
        with patch('asyncio.new_event_loop', return_value=mock_event_loop):
            dir1 = temp_targets / "dir1" / "src"
            dir1.mkdir(parents=True, exist_ok=True)
            result = runner.invoke(main, ['--target', str(dir1), '--force'])
            
            assert result.exit_code == 0
            assert mock_gen.called
            assert mock_gen.call_args[1]['force'] is True

```

### ./tests/test_ignore.py

```
import pytest
from pathlib import Path
from codebase_doc.utils.ignore_parser import IgnorePatternHandler

@pytest.fixture
def ignore_handler() -> IgnorePatternHandler:
    """Create IgnorePatternHandler instance"""
    return IgnorePatternHandler()

@pytest.fixture
def temp_ignore_file(tmp_path: Path) -> Path:
    """Create temporary .docsignore file"""
    ignore_file = tmp_path / ".docsignore"
    ignore_file.write_text("""
# Python files
*.pyc
__pycache__

# Git directory
.git/

# Virtual environments
venv/
.venv/
""")
    return ignore_file

def test_default_patterns(ignore_handler: IgnorePatternHandler) -> None:
    """Test default ignore patterns"""
    assert '__pycache__' in ignore_handler.patterns
    assert '.git' in ignore_handler.patterns
    assert 'venv' in ignore_handler.patterns

def test_load_patterns(
    ignore_handler: IgnorePatternHandler,
    temp_ignore_file: Path
) -> None:
    """Test loading patterns from file"""
    ignore_handler.load_patterns(temp_ignore_file)
    
    assert '*.pyc' in ignore_handler.patterns
    assert '.git/' in ignore_handler.patterns
    assert 'venv/' in ignore_handler.patterns

def test_should_ignore(ignore_handler: IgnorePatternHandler) -> None:
    """Test path ignoring logic"""
    assert ignore_handler.should_ignore('__pycache__/file.pyc')
    assert ignore_handler.should_ignore('.git/config')
    assert not ignore_handler.should_ignore('src/main.py')

def test_reset_patterns(
    ignore_handler: IgnorePatternHandler,
    temp_ignore_file: Path
) -> None:
    """Test pattern reset functionality"""
    ignore_handler.load_patterns(temp_ignore_file)
    ignore_handler.reset_to_defaults()
    
    assert len(ignore_handler.patterns) == len(ignore_handler.DEFAULT_PATTERNS)
    assert all(p in ignore_handler.DEFAULT_PATTERNS for p in ignore_handler.patterns)
```

### ./tests/test_generator.py

```

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import asyncio
import shutil

from codebase_doc.generator import DocumentationGenerator

@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create temporary project structure"""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create test files and directories
    src_dir = project_dir / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('test')")
    
    # Create output directory
    docs_dir = project_dir / "docs"
    docs_dir.mkdir()
    
    return project_dir

@pytest.fixture
def generator(temp_project: Path) -> DocumentationGenerator:
    """Create DocumentationGenerator instance"""
    return DocumentationGenerator(
        project_name="test_project",
        output_dir=temp_project / "docs",
        force=True,
        logger=Mock()
    )

@pytest.mark.asyncio
async def test_documentation_generation(
    generator: DocumentationGenerator,
    temp_project: Path
) -> None:
    """Test documentation generation process"""
    target_dirs = {temp_project / "src"}
    
    with patch('asyncio.create_subprocess_exec') as mock_exec:
        # Mock successful process execution
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.wait = AsyncMock(return_value=None)
        mock_exec.return_value = mock_process
        
        # Create mock output file
        output_file = generator.latest_dir / "test_output.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.touch()
        
        success = await generator.generate(target_dirs)
        assert success

@pytest.mark.asyncio
async def test_concurrent_generation(
    generator: DocumentationGenerator,
    temp_project: Path
) -> None:
    """Test concurrent documentation generation"""
    dirs = ["dir1", "dir2", "dir3"]
    target_dirs = set()
    
    # Create test directories and files
    for dir_name in dirs:
        dir_path = temp_project / dir_name
        dir_path.mkdir()
        (dir_path / "test.py").write_text("print('test')")
        target_dirs.add(dir_path)
    
    with patch('asyncio.create_subprocess_exec') as mock_exec:
        # Mock successful process execution
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.wait = AsyncMock(return_value=None)
        mock_exec.return_value = mock_process
        
        # Create mock output files
        for dir_path in target_dirs:
            output_file = generator.latest_dir / f"{dir_path.name}.md"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.touch()
        
        success = await generator.generate(target_dirs)
        assert success
        assert mock_exec.call_count == len(dirs)
```

### ./cli.py

```
# Location: ./codebase_doc/cli.py

import asyncio
import click
import sys
from pathlib import Path
from typing import List, Set, Optional
from rich.progress import track
from rich import print as rprint

from .utils.logger import setup_logging
from .generator import DocumentationGenerator

def collect_target_directories(targets: List[str], target_files: List[str]) -> Set[Path]:
    """Collect and validate target directories from CLI arguments and files"""
    all_targets = set()
    
    # Process direct target arguments
    for target in targets:
        try:
            path = Path(target)
            if not path.is_absolute():
                path = Path.cwd() / path
            path = path.resolve()
            
            if path.exists() and path.is_dir():
                all_targets.add(path)
            else:
                rprint(f"[red]Error:[/red] Invalid target directory: {target}")
                
        except Exception as e:
            rprint(f"[red]Error processing target {target}:[/red] {str(e)}")
                    
    # Process target files
    for file_path in target_files:
        try:
            file_path = Path(file_path).resolve()
            if not file_path.exists():
                rprint(f"[red]Error:[/red] Target file does not exist: {file_path}")
                continue
                
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        path = Path(line).resolve()
                        if not path.exists():
                            rprint(f"[yellow]Warning:[/yellow] Directory from file does not exist: {line}")
                            continue
                        if not path.is_dir():
                            rprint(f"[yellow]Warning:[/yellow] Path from file is not a directory: {line}")
                            continue
                        
                        all_targets.add(path)
                    except Exception as e:
                        rprint(f"[red]Error processing path {line}:[/red] {str(e)}")
                        
        except Exception as e:
            rprint(f"[red]Error processing target file {file_path}:[/red] {str(e)}")
    
    return all_targets

def get_project_name() -> str:
    """Get project name from current directory"""
    return Path.cwd().name

@click.command()
@click.option('--target', '-t', multiple=True, help='Target directory for documentation generation')
@click.option('--file', '-f', multiple=True, help='File containing target directories')
@click.option('--output-dir', '-o', default='/root/docs', help='Custom output directory')
@click.option('--force', is_flag=True, help='Force overwrite existing documentation')
def main(target: List[str], file: List[str], output_dir: str, force: bool) -> None:
    """Generate documentation for Python projects"""
    # Setup logging
    project_name = get_project_name()
    logger = setup_logging(project_name)
    
    try:
        # Collect target directories
        if not target and not file:
            rprint("[red]Error:[/red] No target directories specified. Use --target or --file")
            sys.exit(1)
        
        targets = collect_target_directories(target, file)
        if not targets:
            rprint("[red]Error:[/red] No valid target directories found")
            sys.exit(1)
        
        # Initialize generator
        generator = DocumentationGenerator(
            project_name=project_name,
            output_dir=Path(output_dir),
            force=force,
            logger=logger
        )
        
        # Run generator in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(generator.generate(targets))
            if not success:
                rprint("[red]Error:[/red] Documentation generation failed")
                sys.exit(1)
        finally:
            loop.close()
            
    except Exception as e:
        rprint(f"[red]Fatal error:[/red] {str(e)}")
        sys.exit(1)
    
    rprint("[green]Documentation generated successfully![/green]")
    sys.exit(0)

if __name__ == '__main__':
    main()
```

### ./utils/processor.py

```
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
import logging

class SequentialProcessor:
    """Sequential documentation processing for improved reliability"""
    
    def __init__(
        self,
        logger: logging.Logger,
        timeout: int = 30,
        retry_delay: int = 2
    ):
        self.logger = logger
        self.timeout = timeout
        self.retry_delay = retry_delay
    
    async def process_folders(
        self,
        folders: List[str],
        docs_dir: Path,
        processor_func: callable
    ) -> List[Tuple[str, bool]]:
        """Process folders sequentially with delay"""
        results = []
        
        for folder in folders:
            try:
                self.logger.info(f"Processing folder: {folder}")
                result = await processor_func(folder, docs_dir)
                results.append(result)
                
                # Add delay between processing
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                self.logger.error(f"Error processing {folder}: {e}")
                results.append((folder, False))
        
        return results
    
    async def cleanup_process(self, pid: int) -> None:
        """Clean up hung process"""
        try:
            import psutil
            process = psutil.Process(pid)
            
            # Try graceful termination first
            process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(),  # type: ignore
                    timeout=5.0
                )
                self.logger.debug(f"Process {pid} terminated gracefully")
                return
            except asyncio.TimeoutError:
                pass
            
            # Force kill if still running
            process.kill()
            await process.wait()  # type: ignore
            self.logger.debug(f"Process {pid} force killed")
            
        except psutil.NoSuchProcess:
            self.logger.debug(f"Process {pid} already terminated")
        except Exception as e:
            self.logger.error(f"Error cleaning up process {pid}: {e}")
```

### ./utils/backup_handler.py

```
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
```

### ./utils/__init__.py

```

```

### ./utils/recovery.py

```
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
```

### ./utils/ignore_parser.py

```
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
```

### ./utils/progress.py

```
from typing import Dict, Optional, Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.console import Console
from rich.table import Table
from datetime import datetime

class ProcessTracker:
    """Tracks and displays documentation generation progress"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.start_time = datetime.now()
        self.tasks: Dict[str, TaskID] = {}
        
        # Initialize progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold cyan]{task.fields[status]}"),
            console=self.console
        )
    
    def add_task(self, directory: str) -> TaskID:
        """Add a new directory task to track"""
        task_id = self.progress.add_task(
            description=f"Processing {directory}",
            total=100,
            status="Starting...",
        )
        self.tasks[directory] = task_id
        return task_id
    
    def update_task(self, directory: str, progress: int, status: str) -> None:
        """Update task progress and status"""
        if directory in self.tasks:
            self.progress.update(
                self.tasks[directory],
                completed=progress,
                status=status
            )
    
    def mark_complete(self, directory: str, success: bool = True) -> None:
        """Mark a task as complete"""
        if directory in self.tasks:
            status = "✅ Complete" if success else "❌ Failed"
            self.progress.update(
                self.tasks[directory],
                completed=100,
                status=status
            )
            
    async def track_process(self, generator: Any) -> None:
        """Start tracking process"""
        with self.progress:
            await generator.generate()
            
    def print_summary(self, results: Dict[str, bool]) -> None:
        """Print final summary table"""
        table = Table(title="Documentation Generation Summary")
        
        table.add_column("Directory", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Duration", style="blue")
        
        total_time = datetime.now() - self.start_time
        success_count = sum(1 for success in results.values() if success)
        
        for directory, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            table.add_row(directory, status, str(total_time))
        
        self.console.print(table)
        self.console.print(f"\nTotal Success Rate: {success_count}/{len(results)}")
```

### ./utils/logger.py

```
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
```

### ./utils/summary.py

```
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
```

