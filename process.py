#!/usr/bin/env python3
"""
PDF Processing Pipeline Runner

This script runs the PDF data extraction and ML model processing pipeline:
1. Executes extract_data_from_pdf.py and waits for completion
2. Executes ml-model.py after the first script completes successfully

Usage:
    python run_pipeline.py

The script will exit with code 1 if any step fails.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_script(script_name, description):
    """
    Run a Python script and wait for it to complete.
    
    Args:
        script_name (str): Name of the Python script to run
        description (str): Description of what the script does
    
    Returns:
        bool: True if script completed successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Starting: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nSUCCESS: {description} completed successfully!")
        print(f"Duration: {duration:.2f} seconds")
        
        # Print stdout if there's any output
        if result.stdout.strip():
            print(f"\nOutput:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nRROR: {description} failed!")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Exit code: {e.returncode}")
        
        if e.stdout:
            print(f"\nOutput:")
            print(e.stdout)
        
        if e.stderr:
            print(f"\nError details:")
            print(e.stderr)
        
        return False
        
    except FileNotFoundError:
        print(f"\nERROR: Script '{script_name}' not found!")
        print(f"Make sure the file exists in the current directory: {os.getcwd()}")
        return False
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nUNEXPECTED ERROR: {description} failed!")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Error: {str(e)}")
        return False

def main():
    """Main pipeline execution function."""
    print("Starting PDF Processing Pipeline")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Check if required scripts exist
    scripts_to_check = ['extract_data_from_pdf.py', 'ml-model.py']
    missing_scripts = []
    
    for script in scripts_to_check:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\nERROR: Missing required scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        print("\nPlease ensure all scripts are in the current directory.")
        sys.exit(1)
    
    total_start_time = time.time()
    
    # Step 1: Run PDF data extraction
    success = run_script(
        'extract_data_from_pdf.py',
        'PDF Data Extraction'
    )
    
    if not success:
        print(f"\nPipeline FAILED at PDF data extraction step!")
        sys.exit(1)
    
    # Step 2: Run ML model processing
    success = run_script(
        'ml-model.py',
        'ML Model Processing'
    )
    
    if not success:
        print(f"\nPipeline FAILED at ML model processing step!")
        sys.exit(1)
    
    # Pipeline completed successfully
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error in pipeline runner: {str(e)}")
        sys.exit(1)
