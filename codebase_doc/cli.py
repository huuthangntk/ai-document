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

# Location: ./codebase_doc/cli.py

import asyncio
import click
import sys
from pathlib import Path
from typing import List, Set, Optional, Tuple
from rich.progress import track
from rich import print as rprint
import os
from contextlib import contextmanager

from .utils.logger import setup_logging
from .generator import DocumentationGenerator

class CLIError(Exception):
    """Custom exception for CLI errors"""
    pass

def validate_directory(path: Path) -> Tuple[bool, Optional[str]]:
    """Validate if a path is a valid directory.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        resolved_path = path.resolve()
        if not resolved_path.exists():
            return False, f"Directory does not exist: {path}"
        if not resolved_path.is_dir():
            return False, f"Path is not a directory: {path}"
        if not os.access(resolved_path, os.R_OK):
            return False, f"No read permission for directory: {path}"
        return True, None
    except Exception as e:
        return False, f"Error validating directory {path}: {str(e)}"

def validate_file(path: Path) -> Tuple[bool, Optional[str]]:
    """Validate if a path is a valid file.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        resolved_path = path.resolve()
        if not resolved_path.exists():
            return False, f"File does not exist: {path}"
        if not resolved_path.is_file():
            return False, f"Path is not a file: {path}"
        if not os.access(resolved_path, os.R_OK):
            return False, f"No read permission for file: {path}"
        return True, None
    except Exception as e:
        return False, f"Error validating file {path}: {str(e)}"

@contextmanager
def error_handling(exit_on_error: bool = True):
    """Context manager for handling CLI errors.
    
    Args:
        exit_on_error: Whether to exit the program on error
    """
    try:
        yield
    except CLIError as e:
        rprint(f"[red]Error:[/red] {str(e)}")
        if exit_on_error:
            sys.exit(1)
    except Exception as e:
        rprint(f"[red]Fatal error:[/red] {str(e)}")
        if exit_on_error:
            sys.exit(1)

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

def collect_target_directories(targets: List[str], target_files: List[str]) -> Set[Path]:
    """Collect and validate target directories from CLI arguments and files.
    
    Args:
        targets: List of target directory paths
        target_files: List of files containing target directories
        
    Returns:
        Set of validated Path objects
        
    Raises:
        CLIError: If no valid targets are found
    """
    all_targets = set()
    errors = []
    
    # Process direct target arguments
    for target in targets:
        try:
            path = Path(target).resolve()
            is_valid, error = validate_directory(path)
            if is_valid:
                all_targets.add(path)
            else:
                errors.append(error)
                
        except Exception as e:
            errors.append(f"Error processing target {target}: {str(e)}")
                    
    # Process target files
    for file_path in target_files:
        try:
            path = Path(file_path)
            is_valid, error = validate_file(path)
            if not is_valid:
                errors.append(error)
                continue
                
            with open(path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        dir_path = Path(line).resolve()
                        is_valid, error = validate_directory(dir_path)
                        if is_valid:
                            all_targets.add(dir_path)
                        else:
                            errors.append(f"Line {line_num}: {error}")
                    except Exception as e:
                        errors.append(f"Error processing line {line_num} in {path}: {str(e)}")
                        
        except Exception as e:
            errors.append(f"Error processing target file {file_path}: {str(e)}")
    
    # Report errors if any
    if errors:
        for error in errors:
            rprint(f"[yellow]Warning:[/yellow] {error}")
    
    if not all_targets:
        raise CLIError("No valid target directories found")
        
    return all_targets

def get_project_name() -> str:
    """Get project name from current directory."""
    try:
        return Path.cwd().name
    except Exception as e:
        raise CLIError(f"Failed to get project name: {str(e)}")

def setup_event_loop() -> asyncio.AbstractEventLoop:
    """Setup and return a new event loop.
    
    Returns:
        Configured event loop
        
    Raises:
        CLIError: If event loop creation fails
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    except Exception as e:
        raise CLIError(f"Failed to setup event loop: {str(e)}")

def run_generator(
    generator: DocumentationGenerator,
    targets: Set[Path],
    loop: asyncio.AbstractEventLoop
) -> bool:
    """Run the documentation generator.
    
    Args:
        generator: Configured DocumentationGenerator instance
        targets: Set of target directories
        loop: Event loop to use
        
    Returns:
        bool: True if generation was successful
        
    Raises:
        CLIError: If generation fails
    """
    try:
        return loop.run_until_complete(generator.generate(targets))
    except Exception as e:
        raise CLIError(f"Documentation generation failed: {str(e)}")

@click.command()
@click.option(
    '--target', '-t',
    multiple=True,
    help='Target directory for documentation generation'
)
@click.option(
    '--file', '-f',
    multiple=True,
    help='File containing target directories'
)
@click.option(
    '--output-dir', '-o',
    default=str(Path.home() / 'docs'),
    help='Custom output directory'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force overwrite existing documentation'
)
def main(target: List[str], file: List[str], output_dir: str, force: bool) -> None:
    """Generate documentation for Python projects."""
    with error_handling():
        # Validate required arguments
        if not target and not file:
            raise CLIError("No target directories specified. Use --target or --file")
        
        # Setup logging
        project_name = get_project_name()
        logger = setup_logging(project_name)
        
        # Collect and validate target directories
        targets = collect_target_directories(list(target), list(file))
        
        # Initialize generator
        generator = DocumentationGenerator(
            project_name=project_name,
            output_dir=Path(output_dir),
            force=force,
            logger=logger
        )
        
        # Setup and run event loop
        loop = setup_event_loop()
        try:
            success = run_generator(generator, targets, loop)
            if not success:
                raise CLIError("Documentation generation failed")
                
            rprint("[green]Documentation generated successfully![/green]")
            sys.exit(0)
        finally:
            loop.close()

if __name__ == '__main__':
    main()