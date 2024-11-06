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