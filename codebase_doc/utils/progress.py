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