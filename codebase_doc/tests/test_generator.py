# Location: ./tests/test_generator.py

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio
import shutil
import os
from typing import AsyncIterator, Dict, Iterator
import logging
from contextlib import asynccontextmanager

from codebase_doc.generator import DocumentationGenerator

class ProcessResult:
    """Track process execution results"""
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.success = False
        self.error = None
        self.output = []
        
    def mark_success(self):
        self.success = True
        
    def mark_failure(self, error: str):
        self.success = False
        self.error = error

class AsyncMockProcess:
    """Enhanced async mock process"""
    def __init__(self, return_code: int = 0, result: ProcessResult = None):
        self.returncode = return_code
        self.result = result
        self._killed = False
        self._stdout = asyncio.Queue()
        self._stderr = asyncio.Queue()
        self.logger = logging.getLogger("test_generator")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def wait(self) -> int:
        """Simulate process execution"""
        await asyncio.sleep(0.1)
        if not self._killed and self.result:
            try:
                self.result.output_file.parent.mkdir(parents=True, exist_ok=True)
                self.result.output_file.write_text("# Generated Documentation\n\nTest content")
                self.result.mark_success()
                await self._feed_output("Successfully generated documentation")
            except Exception as e:
                self.result.mark_failure(str(e))
                await self._feed_output(f"Error: {e}", is_error=True)
        return 0 if not self._killed else 1
    
    async def cleanup(self) -> None:
        """Clean up process resources"""
        if not self._killed:
            self._killed = True
            await self._feed_output("Process cleanup complete")
    
    async def _feed_output(self, message: str, is_error: bool = False):
        """Feed process output"""
        if is_error:
            await self._stderr.put(message.encode())
        else:
            await self._stdout.put(message.encode())
            if self.result:
                self.result.output.append(message)
    
    @property
    def stdout(self):
        return self._stdout
    
    @property
    def stderr(self):
        return self._stderr

@asynccontextmanager
async def create_process_context(output_file: Path):
    """Create process with result tracking"""
    result = ProcessResult(output_file)
    process = AsyncMockProcess(return_code=0, result=result)
    try:
        yield process, result
    finally:
        await process.cleanup()

@pytest.fixture
async def mock_process():
    """Create async mock process fixture"""
    process = AsyncMockProcess(return_code=0)
    async with process:
        yield process

@pytest.fixture
def temp_project(tmp_path: Path) -> Iterator[Path]:
    """Create test project structure"""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test_generator")
    
    try:
        # Create source directory with test files
        src_dir = project_dir / "src"
        src_dir.mkdir()
        
        # Create test files
        (src_dir / "main.py").write_text("""
def main():
    print('test')

if __name__ == '__main__':
    main()
""")
        (src_dir / "__init__.py").write_text("# Test module")
        
        # Create docs directory
        docs_dir = project_dir / "docs"
        docs_dir.mkdir()
        
        yield project_dir
    finally:
        # Cleanup
        try:
            shutil.rmtree(project_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Failed to cleanup {project_dir}: {e}")

@pytest.fixture
def generator(temp_project: Path) -> DocumentationGenerator:
    """Create generator instance with clean state"""
    logger = Mock()
    
    gen = DocumentationGenerator(
        project_name="test_project",
        output_dir=temp_project / "docs",
        force=True,
        logger=logger,
        max_concurrent=5,
        timeout=30
    )
    
    # Ensure clean state
    if gen.latest_dir.exists():
        shutil.rmtree(gen.latest_dir)
    gen.latest_dir.mkdir(parents=True)
    
    return gen

async def mock_summary_generate(*args, **kwargs):
    """Mock successful summary generation"""
    return True

@pytest.mark.asyncio
async def test_documentation_generation(
    generator: DocumentationGenerator,
    temp_project: Path
) -> None:
    """Test basic documentation generation process with enhanced tracking"""
    target_dirs = {temp_project / "src"}
    src_dir = temp_project / "src"
    process_results: Dict[Path, ProcessResult] = {}
    
    # Ensure target directory exists and has content
    assert src_dir.exists(), "Source directory not created"
    assert list(src_dir.glob('*.py')), "No Python files in source directory"
    
    async def create_mock_subprocess(*args, **kwargs):
        """Create process with result tracking"""
        # Parse command arguments
        cmd_args = args[1:]  # Skip 'cdigest'
        input_dir = next(arg for arg in cmd_args if not arg.startswith('-'))
        input_path = Path(input_dir)
        
        # Setup output file and result tracking
        output_file = generator.latest_dir / f"{input_path.name}.md"
        result = ProcessResult(output_file)
        process_results[input_path] = result
        
        # Create and return process
        return AsyncMockProcess(return_code=0, result=result)
    
    # Setup output directory
    output_file = generator.latest_dir / "src.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with patch('asyncio.create_subprocess_exec',
               side_effect=create_mock_subprocess), \
         patch('codebase_doc.utils.summary.SummaryGenerator.generate',
               side_effect=mock_summary_generate):
        
        # Run generation
        success = await generator.generate(target_dirs)
        
        # Verify process results
        assert process_results, "No processes were created"
        for path, result in process_results.items():
            assert result.success, f"Process failed for {path}: {result.error}"
            assert result.output_file.exists(), f"Output file missing for {path}"
            assert "Generated Documentation" in result.output_file.read_text(), \
                   f"Invalid content in {result.output_file}"
        
        # Verify overall success
        assert success, "Documentation generation failed"
        assert output_file.exists(), "Output file not created"
        content = output_file.read_text()
        assert "Generated Documentation" in content, "Invalid output content"
        
        # Verify no leftover processes
        assert all(not result.error for result in process_results.values()), \
               "Some processes had errors"

async def create_mock_subprocess(*args, **kwargs):
    """Helper to create mock subprocess with proper async behavior"""
    process = AsyncMockProcess(return_code=0)
    await process._feed_output()
    return process
# Location: ./tests/test_generator.py (continued)

@pytest.mark.asyncio
async def test_concurrent_generation(
    generator: DocumentationGenerator,
    temp_project: Path
) -> None:
    """Test concurrent generation with enhanced tracking"""
    # Create test directories
    dirs = ["dir1", "dir2", "dir3"]
    target_dirs = set()
    processes: Dict[Path, ProcessResult] = {}
    
    for dir_name in dirs:
        dir_path = temp_project / dir_name
        dir_path.mkdir()
        (dir_path / "test.py").write_text("print('test')")
        target_dirs.add(dir_path)
    
    async def create_mock_subprocess(*args, **kwargs):
        """Create process with result tracking"""
        cmd_args = args[1:]  # Skip 'cdigest'
        input_dir = next(arg for arg in cmd_args if not arg.startswith('-'))
        input_path = Path(input_dir)
        output_file = generator.latest_dir / f"{input_path.name}.md"
        
        # Create and track process
        result = ProcessResult(output_file)
        processes[input_path] = result
        return AsyncMockProcess(return_code=0, result=result)
    
    with patch('asyncio.create_subprocess_exec', 
               side_effect=create_mock_subprocess), \
         patch('codebase_doc.utils.summary.SummaryGenerator.generate',
               side_effect=mock_summary_generate):
        
        # Run concurrent generation
        success = await generator.generate(target_dirs)
        
        # Verify results
        assert success, "Concurrent generation failed"
        assert len(processes) == len(dirs), "Not all processes created"
        
        # Check each process result
        for dir_path, result in processes.items():
            assert result.success, f"Process failed for {dir_path}: {result.error}"
            assert result.output_file.exists(), f"Output file missing for {dir_path}"
            content = result.output_file.read_text()
            assert "Generated Documentation" in content

@pytest.mark.asyncio
async def test_backup_handling(
    generator: DocumentationGenerator,
    temp_project: Path
) -> None:
    """Test backup creation with enhanced tracking"""
    src_dir = temp_project / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # Create existing documentation
    existing_doc = generator.latest_dir / "existing.md"
    existing_doc.parent.mkdir(parents=True, exist_ok=True)
    existing_doc.write_text("# Existing documentation")
    
    # Setup process tracking
    process_results: Dict[Path, ProcessResult] = {}
    
    async def create_mock_subprocess(*args, **kwargs):
        """Create process with backup handling"""
        cmd_args = args[1:]
        input_dir = next(arg for arg in cmd_args if not arg.startswith('-'))
        input_path = Path(input_dir)
        output_file = generator.latest_dir / f"{input_path.name}.md"
        
        # Create and track process
        result = ProcessResult(output_file)
        process_results[input_path] = result
        return AsyncMockProcess(return_code=0, result=result)
    
    with patch('asyncio.create_subprocess_exec',
               side_effect=create_mock_subprocess), \
         patch('codebase_doc.utils.summary.SummaryGenerator.generate',
               side_effect=mock_summary_generate):
        
        # Run generation without force flag
        generator.force = False
        success = await generator.generate({src_dir})
        
        # Verify results
        assert success, "Backup handling failed"
        
        # Check backup creation
        backup_dirs = list(generator.backup_dir.glob("docs_backup_*"))
        assert len(backup_dirs) > 0, "No backup directory created"
        
        # Verify backup content
        backup_files = list(backup_dirs[0].glob("**/*.md"))
        assert len(backup_files) > 0, "No files in backup"
        backup_content = backup_files[0].read_text()
        assert "Existing documentation" in backup_content
        
        # Check process results
        for path, result in process_results.items():
            assert result.success, f"Process failed for {path}: {result.error}"
            assert result.output_file.exists(), f"Output missing for {path}"

def test_output_path_generation(
    generator: DocumentationGenerator,
    temp_project: Path
):
    """Test output path generation logic"""
    test_paths = [
        Path("/test/path/dir"),
        Path("relative/path/dir"),
        Path("./current/dir"),
        Path("../parent/dir")
    ]
    
    for input_path in test_paths:
        output_path = generator._get_output_path(input_path)
        # Convert to relative path if absolute
        if output_path.is_absolute():
            try:
                output_path = output_path.relative_to(generator.latest_dir)
            except ValueError:
                rel_path = output_path.parts[-1]
                output_path = Path(rel_path).with_suffix('.md')
                
        assert not output_path.is_absolute(), f"Path should be relative: {output_path}"
        assert output_path.suffix == '.md', f"Wrong extension: {output_path}"

@pytest.mark.asyncio
async def test_error_handling(
    generator: DocumentationGenerator,
    temp_project: Path
) -> None:
    """Test error handling scenarios"""
    logger = logging.getLogger("test_generator")
    src_dir = temp_project / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created test directory: {src_dir}")
    
    error_cases = [
        (1, False, "Process failure"),
        (-9, False, "Kill signal"),
        (None, False, "Timeout")
    ]
    
    for return_code, expected_success, scenario in error_cases:
        logger.debug(f"Testing scenario: {scenario}")
        
        async def mock_error_process(*args, **kwargs):
            process = AsyncMockProcess(return_code=return_code)
            await process._feed_output()
            return process
        
        with patch('asyncio.create_subprocess_exec', side_effect=mock_error_process):
            success = await generator.generate({src_dir})
            assert success == expected_success, f"Failed for {scenario}"
            logger.debug(f"Completed {scenario} with success={success}")

def test_output_path_generation(generator: DocumentationGenerator):
    """Test output path generation logic"""
    logger = logging.getLogger("test_generator")
    
    test_paths = [
        Path("/test/path/dir"),
        Path("relative/path/dir"),
        Path("./current/dir"),
        Path("../parent/dir")
    ]
    
    for input_path in test_paths:
        logger.debug(f"Testing path: {input_path}")
        output_path = generator._get_output_path(input_path)
        
        # Ensure relative path
        if output_path.is_absolute():
            try:
                output_path = output_path.relative_to(generator.latest_dir)
            except ValueError:
                output_path = Path(output_path.name)
        
        logger.debug(f"Generated output path: {output_path}")
        assert not output_path.is_absolute(), f"Path should be relative: {output_path}"
        assert output_path.suffix == '.md', f"Wrong extension: {output_path}"

if __name__ == '__main__':
    pytest.main([__file__, "-v"])