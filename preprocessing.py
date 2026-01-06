"""
Data Preprocessing Utilities for Hard Drive RUL Prediction
============================================================
This module contains all data preprocessing functions for the streaming ML pipeline.
"""

import polars as pl
import os
import glob
import math
from typing import List, Set
import config


def get_file_list(folder: str) -> List[str]:
    """
    Retrieves a sorted list of CSV files from the specified folder.
    
    Args:
        folder: Path to the folder containing CSV files
        
    Returns:
        Sorted list of file paths
        
    Raises:
        FileNotFoundError: If no CSV files are found in the folder
    """
    pattern = os.path.join(folder, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return files


def find_failed_serials(file_list: List[str], verbose: bool = True) -> List[str]:
    """
    Step 1: Identify all unique serial numbers of drives that failed.
    
    This function scans all CSV files and collects serial numbers where failure=1.
    
    Args:
        file_list: List of CSV file paths to process
        verbose: Whether to print progress messages
        
    Returns:
        List of unique serial numbers that experienced failure
    """
    if verbose:
        print("\n=== STEP 1: Identifying Failed Drives ===")
    
    failed_serials = set()
    
    for i, file_path in enumerate(file_list):
        try:
            # Read only necessary columns for memory efficiency
            df = pl.read_csv(
                file_path, 
                columns=["serial_number", "failure"], 
                ignore_errors=True,
                n_threads=config.PREPROCESSING_CONFIG["n_threads"]
            )
            
            # Filter for failures (cast to Int64 for safety)
            df = df.with_columns(
                pl.col("failure").cast(pl.Int64, strict=False).fill_null(0)
            )
            failures = df.filter(pl.col("failure") == 1)
            
            # Add to set (automatically handles duplicates)
            current_failures = failures["serial_number"].unique().to_list()
            failed_serials.update(current_failures)
            
            # Report progress
            if verbose and ((i + 1) % config.PREPROCESSING_CONFIG["batch_size"] == 0 
                            or (i + 1) == len(file_list)):
                print(f"   Processed {i + 1}/{len(file_list)} files... "
                      f"({len(failed_serials)} unique failures found)")
                
        except Exception as e:
            if verbose:
                print(f"   [WARNING] Error reading {os.path.basename(file_path)}: {e}")
            continue

    if verbose:
        print(f"✓ Total unique failed drives found: {len(failed_serials)}\n")
    
    return list(failed_serials)


def extract_history(file_list: List[str], target_serials: List[str], 
                   verbose: bool = True) -> pl.DataFrame:
    """
    Step 2: Extract complete history for the specified serial numbers.
    
    This function reads all CSV files and extracts rows for drives that failed,
    including all their historical SMART data leading up to failure.
    
    Args:
        file_list: List of CSV file paths to process
        target_serials: List of serial numbers to extract
        verbose: Whether to print progress messages
        
    Returns:
        Polars DataFrame with complete history of target drives
    """
    if verbose:
        print("=== STEP 2: Extracting History for Failed Drives ===")
    
    dataframes_list = []
    base_cols = config.BASE_COLUMNS
    
    for i, file_path in enumerate(file_list):
        try:
            # Dynamically detect SMART columns
            schema = pl.scan_csv(file_path).collect_schema().names()
            smart_cols = [c for c in schema if "smart" in c.lower() and "raw" in c.lower()]
            cols_to_read = [c for c in base_cols + smart_cols if c in schema]
            
            # Read file with specific columns
            df = pl.read_csv(
                file_path, 
                columns=cols_to_read,
                ignore_errors=True
            )
            
            # Filter for target serial numbers
            df_filtered = df.filter(pl.col("serial_number").is_in(target_serials))
            
            if not df_filtered.is_empty():
                # Normalize data types to prevent concatenation errors
                smart_cols_present = [c for c in smart_cols if c in df_filtered.columns]
                exprs = [
                    pl.col(c).cast(pl.Float64, strict=False).fill_null(0).alias(c) 
                    for c in smart_cols_present
                ]
                df_filtered = df_filtered.with_columns(exprs)
                df_filtered = df_filtered.with_columns(pl.col("date").cast(pl.String))
                
                dataframes_list.append(df_filtered)

            if verbose and ((i + 1) % config.PREPROCESSING_CONFIG["batch_size"] == 0 
                            or (i + 1) == len(file_list)):
                print(f"   Processed {i + 1}/{len(file_list)} files. "
                      f"Accumulated {len(dataframes_list)} fragments.")

        except Exception as e:
            if verbose:
                print(f"   [WARNING] Error processing {os.path.basename(file_path)}: {e}")

    if verbose:
        print("   Concatenating all fragments...")
    
    # Concatenate with diagonal_relaxed to handle varying columns
    full_df = pl.concat(dataframes_list, how="diagonal_relaxed")
    full_df = full_df.fill_null(0)
    
    if verbose:
        print(f"✓ Extracted {len(full_df):,} records\n")
    
    return full_df


def calculate_rul(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """
    Step 3: Calculate Remaining Useful Life (RUL) for each record.
    
    RUL = (Failure Date - Current Date) in days
    
    Args:
        df: DataFrame with drive history
        verbose: Whether to print progress messages
        
    Returns:
        DataFrame with RUL column added
    """
    if verbose:
        print("=== STEP 3: Calculating RUL ===")
    
    # Convert date to proper Date type
    df = df.with_columns(pl.col("date").str.to_date(strict=False))
    
    # Calculate max date (failure date) per serial number
    df_rul = df.with_columns(
        max_date = pl.col("date").max().over("serial_number")
    ).with_columns(
        RUL = (pl.col("max_date") - pl.col("date")).dt.total_days()
    ).drop("max_date")
    
    # Remove any negative RULs (data errors)
    df_rul = df_rul.filter(pl.col("RUL") >= 0)
    
    # Sort by date (CRITICAL for streaming simulation)
    df_rul = df_rul.sort("date")
    
    if verbose:
        print(f"✓ Final dataset shape: {df_rul.shape[0]:,} rows × {df_rul.shape[1]} columns")
        print(f"   RUL range: {df_rul['RUL'].min()} to {df_rul['RUL'].max()} days\n")
    
    return df_rul


def preprocess_data(data_folder: str = None, 
                   output_file: str = None,
                   verbose: bool = True) -> str:
    """
    Complete preprocessing pipeline: Find failures → Extract history → Calculate RUL → Save
    
    Args:
        data_folder: Path to folder containing daily CSV files
        output_file: Path where preprocessed data will be saved
        verbose: Whether to print progress messages
        
    Returns:
        Path to the saved preprocessed file
    """
    if data_folder is None:
        data_folder = config.DEFAULT_DATA_FOLDER
    if output_file is None:
        output_file = config.PREPROCESSED_FILE
        
    if verbose:
        print(f"\n{'='*60}")
        print(f"PREPROCESSING PIPELINE")
        print(f"{'='*60}")
        print(f"Source: {data_folder}")
        print(f"Output: {output_file}")
        print(f"{'='*60}\n")
    
    try:
        # Step 1: Get file list
        files = get_file_list(data_folder)
        if verbose:
            print(f"Found {len(files)} CSV files\n")
        
        # Step 2: Find failed drives
        target_serials = find_failed_serials(files, verbose)
        
        if not target_serials:
            raise ValueError("No failed serial numbers found in data")
        
        # Step 3: Extract history
        df = extract_history(files, target_serials, verbose)
        
        # Step 4: Calculate RUL
        df_final = calculate_rul(df, verbose)
        
        # Step 5: Save to CSV
        if verbose:
            print(f"=== Saving to {output_file} ===")
        df_final.write_csv(output_file)
        if verbose:
            print(f"✓ Success! Data saved to: {output_file}\n")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        raise


def apply_feature_transformation(x: dict, method: str = "log") -> dict:
    """
    Apply transformation to feature dictionary (for streaming pipeline).
    
    Args:
        x: Dictionary of features (from River stream)
        method: Transformation method ("raw", "log", "normalized")
        
    Returns:
        Transformed feature dictionary
    """
    if method == "raw":
        # No transformation, just ensure all values are float
        return {k: float(v) for k, v in x.items()}
    
    elif method == "log":
        # Apply log(1 + x) transformation to handle large SMART values
        return {k: math.log1p(float(v)) for k, v in x.items()}
    
    elif method == "normalized":
        # Normalization handled by StandardScaler in model pipeline
        return {k: float(v) for k, v in x.items()}
    
    else:
        raise ValueError(f"Unknown transformation method: {method}")


if __name__ == "__main__":
    """
    Stand-alone execution: Run the preprocessing pipeline
    """
    import sys
    
    data_folder = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_DATA_FOLDER
    output_file = sys.argv[2] if len(sys.argv) > 2 else config.PREPROCESSED_FILE
    
    preprocess_data(data_folder, output_file, verbose=True)
