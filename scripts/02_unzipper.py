#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path
import sys

def unzip_data(zip_file_path):
    """
    Unzips a file to a folder with the same name in the same directory.
    
    Args:
        zip_file_path (str/Path): Path to the zip file
    """
    zip_path = Path(zip_file_path).resolve()  # Convert to absolute path
    
    if not zip_path.exists():
        print(f"Error: Zip file not found: {zip_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory named after the zip file (without .zip extension)
    extract_to = zip_path.parent / zip_path.stem
    
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {zip_path.name} to {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"Successfully extracted to {extract_to}")
    except zipfile.BadZipFile:
        print(f"Error: The file is not a valid zip file: {zip_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Unzip utility that extracts to same directory as zip file',
        usage='python unzip_data.py ZIP_FILE'
    )
    parser.add_argument(
        'zip_file',
        help='Path to the zip file (relative or absolute)'
    )
    
    args = parser.parse_args()
    unzip_data(args.zip_file)

if __name__ == "__main__":
    main()