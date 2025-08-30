
# Logging utilities for PQC Biometric System
# Redirects verbose output to files while keeping terminal clean


import os
import sys
import builtins
from datetime import datetime
from pathlib import Path

# Store reference to original print before any global overrides
_original_print = getattr(builtins, '_original_print', print)

class ComponentLogger:
    """Logger for individual system components"""
    
    def __init__(self, component_name, log_dir="logs"):
        self.component_name = component_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # CreateS log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{component_name}_{timestamp}.log"
        
        # StoreS original print function
        self.file_handle = None
        self.original_builtin_print = None
    
    def __enter__(self):
        """Start logging to file"""
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        
        # Writes header using original print to bypass global override
        _original_print(f"=== {self.component_name.upper()} LOG ===", file=self.file_handle)
        _original_print(f"Timestamp: {datetime.now()}", file=self.file_handle)
        _original_print("="*50, file=self.file_handle)
        _original_print("", file=self.file_handle)
        self.file_handle.flush()
        
        # Overrides print to redirect to file using original print
        original_builtin_print = builtins.print
        def log_print(*args, **kwargs):
            # Forces output to our log file using original print
            kwargs['file'] = self.file_handle
            _original_print(*args, **kwargs)
            self.file_handle.flush()
        
        builtins.print = log_print
        self.original_builtin_print = original_builtin_print
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop logging and restore stdout"""
        if self.file_handle:
            _original_print("\n\n=== LOG COMPLETE ===", file=self.file_handle)
            self.file_handle.close()
        
        # Restores original print function
        builtins.print = self.original_builtin_print
        
        # Uses original print for terminal message (bypasses global override)
        _original_print(f"[LOG] {self.component_name} details logged to: {self.log_file}")
    
    def terminal_print(self, message):
        """Print message to terminal even when logging to file"""
        # Uses original print to bypass global override and show on terminal
        _original_print(message)

def create_summary_log(results_dict, filename="system_summary.log"):
    """Create a summary log file with key results"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    summary_file = log_dir / filename
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== PQC BIOMETRIC SYSTEM SUMMARY ===\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*40 + "\n\n")
        
        for component, results in results_dict.items():
            f.write(f"{component.upper()}:\n")
            if isinstance(results, dict):
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {results}\n")
            f.write("\n")
    
    # Uses original print to bypass global override
    _original_print(f"[log] System summary saved to: {summary_file}")
    return summary_file
