# Location: ./codebase_doc/tests/test_cli.py

import asyncio
import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from codebase_doc.cli import main

@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner with isolated filesystem"""
    return CliRunner(mix_stderr=False)

@pytest.fixture
def temp_targets(tmp_path: Path) -> Path:
    """Create temporary target directories and files with proper structure"""
    # Create test directories with content
    dir1 = tmp_path / "dir1" / "src"
    dir2 = tmp_path / "dir2" / "src"
    
    for d in [dir1, dir2]:
        d.mkdir(parents=True)
        (d / "test.py").write_text("print('test')")
    
    # Create target file with proper paths
    target_file = tmp_path / "targets.txt"
    target_file.write_text(f"{dir1}\n{dir2}")
    
    return tmp_path

@pytest.fixture
def mock_event_loop():
    """Create mock event loop with proper async support"""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.run_until_complete = AsyncMock(return_value=True)
    loop.close = MagicMock()
    return loop

def test_cli_help(runner: CliRunner) -> None:
    """Test help output shows all options"""
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert '--target' in result.output
    assert '--file' in result.output
    assert '--force' in result.output
    assert '--output-dir' in result.output

def test_cli_target_option(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test target directory option with proper mocks"""
    with patch('codebase_doc.cli.DocumentationGenerator') as mock_gen, \
         patch('asyncio.new_event_loop', return_value=mock_event_loop):
        # Setup mock generator with proper async behavior
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=True)
        
        # Create test directory with content
        dir1 = temp_targets / "dir1" / "src"
        assert dir1.exists(), "Test directory not created properly"
        
        # Run CLI command
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['--target', str(dir1)])
            
            # Verify results
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert mock_gen.called, "Generator was not initialized"
            assert instance.generate.called, "Generator.generate was not called"
            
            # Verify generator initialization
            assert mock_gen.call_args is not None
            kwargs = mock_gen.call_args[1]
            assert kwargs['force'] is False
            assert isinstance(kwargs['output_dir'], Path)

def test_cli_file_option(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test file option with proper mocks"""
    with patch('codebase_doc.cli.DocumentationGenerator') as mock_gen, \
         patch('asyncio.new_event_loop', return_value=mock_event_loop):
        # Setup mock generator
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=True)
        
        # Verify target file exists
        target_file = temp_targets / "targets.txt"
        assert target_file.exists(), "Target file not created properly"
        
        # Run CLI command
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['--file', str(target_file)])
            
            # Verify results
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert mock_gen.called, "Generator was not initialized"
            assert instance.generate.called, "Generator.generate was not called"
            
            # Verify target collection
            mock_gen_instance = mock_gen.return_value
            generate_calls = mock_gen_instance.generate.call_args_list
            assert len(generate_calls) > 0, "No calls to generate()"

def test_cli_force_flag(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test force flag properly passed to generator"""
    with patch('codebase_doc.cli.DocumentationGenerator') as mock_gen, \
         patch('asyncio.new_event_loop', return_value=mock_event_loop):
        # Setup mock generator
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=True)
        
        # Create test directory
        dir1 = temp_targets / "dir1" / "src"
        dir1.mkdir(parents=True, exist_ok=True)
        
        # Run CLI command with force flag
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['--target', str(dir1), '--force'])
            
            # Verify results
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert mock_gen.called, "Generator was not initialized"
            
            # Verify force flag was passed
            kwargs = mock_gen.call_args[1]
            assert kwargs['force'] is True, "Force flag not properly passed"

def test_cli_multiple_targets(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test handling multiple target directories"""
    with patch('codebase_doc.cli.DocumentationGenerator') as mock_gen, \
         patch('asyncio.new_event_loop', return_value=mock_event_loop):
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=True)
        
        # Create multiple test directories
        dir1 = temp_targets / "dir1" / "src"
        dir2 = temp_targets / "dir2" / "src"
        
        # Run CLI command with multiple targets
        with runner.isolated_filesystem():
            result = runner.invoke(main, [
                '--target', str(dir1),
                '--target', str(dir2)
            ])
            
            assert result.exit_code == 0
            assert mock_gen.called
            
            # Verify all targets were processed
            generate_call = instance.generate.call_args
            assert generate_call is not None
            targets = generate_call[0][0]  # First positional arg
            assert len(targets) == 2

def test_cli_error_handling(runner: CliRunner, temp_targets: Path, mock_event_loop):
    """Test CLI error handling for various scenarios"""
    with patch('codebase_doc.cli.DocumentationGenerator') as mock_gen, \
         patch('asyncio.new_event_loop', return_value=mock_event_loop):
        instance = mock_gen.return_value
        instance.generate = AsyncMock(return_value=False)  # Simulate failure
        
        # Test with non-existent directory
        result = runner.invoke(main, ['--target', '/nonexistent/path'])
        assert result.exit_code == 1
        assert 'Error' in result.output