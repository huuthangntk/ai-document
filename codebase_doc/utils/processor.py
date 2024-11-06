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