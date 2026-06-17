#!/usr/bin/env python3
"""
Download and convert ImmunoStates reference matrix from R format to CSV.

This script:
1. Downloads the immunoStatesMatrix.rda file from GitHub OR uses a local file
2. Converts it from R .rda/.RData format to CSV using rpy2
3. Saves it in the appropriate location for metaintegrator

Requirements:
    pip install rpy2
    R must be installed with the immunoStates package

Usage:
    # Download from GitHub (default):
    python download_and_convert_immunostates.py
    
    # Use a local .rda/.RData file:
    python download_and_convert_immunostates.py --input /path/to/file.rda
    
    # With custom output path:
    python download_and_convert_immunostates.py --input /path/to/file.rda --output /path/to/output.csv
    
    # For complex R objects (lists/S3 objects), list components:
    python download_and_convert_immunostates.py --input /path/to/file.rda --inspect
    
    # Extract a specific component from a complex object:
    python download_and_convert_immunostates.py --input /path/to/file.rda --component originalData
    
    # Save complex object as JSON:
    python download_and_convert_immunostates.py --input /path/to/file.rda --output data.json
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
import tempfile
import os
import json

def download_immunostates_rda(output_path: str = None) -> str:
    """
    Download the immunoStates .rda file from GitHub.
    
    Args:
        output_path: Where to save the file (default: temp directory)
        
    Returns:
        Path to downloaded file
    """
    url = "https://github.com/Franenco/ImmunoStates/raw/master/data/immunoStatesMatrix.rda"
    
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / "immunoStatesMatrix.rda"
    else:
        output_path = Path(output_path)
    
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    urlretrieve(url, output_path)
    print("✓ Download complete")
    
    return str(output_path)


def inspect_r_object(r_obj, object_name: str) -> list:
    """
    Inspect and display the structure of an R object.
    
    Args:
        r_obj: The R object to inspect
        object_name: Name of the R object
        
    Returns:
        List of component names (if it's a list), otherwise empty list
    """
    import rpy2.robjects as ro
    
    print("\n" + "="*70)
    print(f"INSPECTING R OBJECT: {object_name}")
    print("="*70)
    
    # Get class
    r_class = ro.r(f'class({object_name})')[0]
    print(f"Class: {r_class}")
    
    # Get structure
    print(f"\nStructure:")
    try:
        structure = ro.r(f'str({object_name})')
        print(structure)
    except:
        print("  (Could not display structure)")
    
    component_names = []
    
    # For lists, show names
    try:
        if r_class in ['list'] or hasattr(r_obj, 'names'):
            names = ro.r(f'names({object_name})')
            if names is not ro.NULL:
                component_names = list(names)
                print(f"\nList components ({len(names)}):")
                for i, name in enumerate(names, 1):
                    # Try to get type and size of each component
                    try:
                        comp_class = ro.r(f'class({object_name}${name})')[0]
                        comp_len = ro.r(f'length({object_name}${name})')[0]
                        print(f"  {i}. {name} - class: {comp_class}, length: {comp_len}")
                    except:
                        print(f"  {i}. {name}")
    except:
        pass
    
    print("\n" + "="*70)
    if component_names:
        print("To extract a specific component, use one of:")
        for name in component_names:
            print(f"  --component {name}")
    print("To save as JSON, use: --output file.json")
    print("="*70)
    
    return component_names


def r_to_python_recursive(r_obj, max_depth=10, current_depth=0):
    """
    Recursively convert R objects to Python objects.
    
    Args:
        r_obj: R object to convert
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        Python object (dict, list, or basic types)
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.rlike.container import TaggedList
    import pandas as pd
    import numpy as np
    
    if current_depth >= max_depth:
        return str(r_obj)
    
    # Handle NULL
    if r_obj is ro.NULL:
        return None
    
    # Handle vectors (atomic types)
    if isinstance(r_obj, ro.vectors.IntVector):
        return list(r_obj)
    elif isinstance(r_obj, ro.vectors.FloatVector):
        return list(r_obj)
    elif isinstance(r_obj, ro.vectors.StrVector):
        return list(r_obj)
    elif isinstance(r_obj, ro.vectors.BoolVector):
        return list(r_obj)
    
    # Handle matrices/arrays
    elif isinstance(r_obj, ro.vectors.Matrix):
        with localconverter(ro.default_converter + pandas2ri.converter):
            np_array = ro.conversion.rpy2py(r_obj)
        return np_array.tolist()
    
    # Handle data.frames
    elif isinstance(r_obj, ro.vectors.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(r_obj)
        return df.to_dict(orient='list')
    
    # Handle lists (including S3 objects)
    elif isinstance(r_obj, (ro.vectors.ListVector, TaggedList)):
        result = {}
        if hasattr(r_obj, 'names') and r_obj.names is not ro.NULL:
            # Named list
            for name in r_obj.names:
                try:
                    result[name] = r_to_python_recursive(
                        r_obj.rx2(name), max_depth, current_depth + 1
                    )
                except:
                    result[name] = str(r_obj.rx2(name))
        else:
            # Unnamed list
            return [r_to_python_recursive(item, max_depth, current_depth + 1) 
                    for item in r_obj]
        
        # Add class attribute if it's an S3 object (for informational purposes)
        if hasattr(r_obj, 'rclass'):
            result['_r_class'] = list(r_obj.rclass)
        
        return result
    
    # Fallback to string representation
    else:
        return str(r_obj)


def convert_rda_to_csv(rda_path: str, csv_path: str, variable_name: str = None, 
                       component: str = None, inspect_only: bool = False) -> None:
    """
    Convert R .rda/.RData file to CSV using rpy2.
    
    Args:
        rda_path: Path to .rda or .RData file
        csv_path: Path to save CSV or JSON file
        variable_name: Optional R object name to extract (auto-detected if None)
        component: Optional component name to extract from a list/S3 object (e.g., "originalData")
        inspect_only: If True, only inspect the object without converting
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        import pandas as pd
        import numpy as np
        
        print(f"Loading R data from: {rda_path}")
        
        # Expand user path if it contains ~
        rda_path = os.path.expanduser(rda_path)
        
        # Verify file exists
        if not os.path.exists(rda_path):
            raise FileNotFoundError(f"Input file not found: {rda_path}")
        
        # Load the .rda or .RData file
        ro.r(f'load("{rda_path}")')
        
        # List all objects in R environment
        r_objects = list(ro.r('ls()'))
        print(f"Found {len(r_objects)} R object(s): {r_objects}")
        
        # Determine which object to use
        if variable_name:
            # Use specified variable
            if variable_name not in r_objects:
                raise ValueError(
                    f"Variable '{variable_name}' not found in .rda file. "
                    f"Available objects: {r_objects}"
                )
            object_name = variable_name
        elif 'immunoStatesMatrix' in r_objects:
            # Default to immunoStatesMatrix if available
            object_name = 'immunoStatesMatrix'
        elif len(r_objects) == 1:
            # Use the only object if there's just one
            object_name = r_objects[0]
            print(f"Auto-detected single object: {object_name}")
        else:
            # Multiple objects, try to be smart
            raise ValueError(
                f"Multiple R objects found: {r_objects}. "
                "Please specify which one to use with --variable option."
            )
        
        print(f"Extracting R object: {object_name}")
        
        # Get the R object
        try:
            r_obj = ro.r[object_name]
        except Exception as e:
            raise ValueError(
                f"Could not extract '{object_name}' from .rda file: {e}"
            )
        
        # Get R object type
        r_class = ro.r(f'class({object_name})')[0]
        print(f"R object class: {r_class}")
        
        # Inspect only mode
        if inspect_only:
            inspect_r_object(r_obj, object_name)
            return
        
        # If component is specified, validate and extract it
        if component:
            # Strip leading $ if present
            component_clean = component.lstrip('$')
            
            # Check if this is a special Python-added field
            if component_clean in ['_r_class']:
                print(f"\nERROR: '{component_clean}' is not an actual R component.")
                print("This is metadata added during Python conversion.")
                print("\nTo see actual R components, run with --inspect flag")
                sys.exit(1)
            
            # Get available component names
            print(f"Extracting component: {component_clean}")
            
            # First verify the component exists
            try:
                names = ro.r(f'names({object_name})')
                if names is not ro.NULL:
                    available_components = list(names)
                    if component_clean not in available_components:
                        print(f"\nERROR: Component '{component_clean}' not found in R object.")
                        print(f"Available components: {available_components}")
                        print("\nRun with --inspect to see component details")
                        sys.exit(1)
            except:
                print(f"WARNING: Could not list components. Attempting extraction anyway...")
            
            # Extract the component
            try:
                # Use safe indexing with [[
                r_obj = ro.r(f'{object_name}[["{component_clean}"]]')
                object_name = f"{object_name}${component_clean}"
                r_class = ro.r(f'class({object_name})')[0]
                print(f"Component class: {r_class}")
            except Exception as e:
                raise ValueError(f"Could not extract component '{component_clean}': {e}")
        
        # Convert to pandas DataFrame based on R object type
        df = None
        output_format = Path(csv_path).suffix.lower()
        
        if r_class in ['matrix', 'array']:
            print("Converting R matrix/array to pandas DataFrame...")
            
            # Get row and column names from R
            rownames = ro.r(f'rownames({object_name})')
            colnames = ro.r(f'colnames({object_name})')
            
            # Convert with pandas context
            with localconverter(ro.default_converter + pandas2ri.converter):
                # Convert to numpy array first
                np_array = ro.conversion.rpy2py(r_obj)
            
            # Create DataFrame with proper row and column names
            if rownames is not ro.NULL:
                rownames = list(rownames)
            else:
                rownames = None
                
            if colnames is not ro.NULL:
                colnames = list(colnames)
            else:
                colnames = None
            
            df = pd.DataFrame(np_array, index=rownames, columns=colnames)
            print(f"✓ Converted matrix to DataFrame")
            
        elif r_class == 'data.frame':
            print("Converting R data.frame to pandas DataFrame...")
            
            # Use pandas converter for data.frame
            with localconverter(ro.default_converter + pandas2ri.converter):
                df = ro.conversion.rpy2py(r_obj)
            
            print(f"✓ Converted data.frame to DataFrame")
            
        elif r_class == 'list' or output_format == '.json':
            print("Converting R object to JSON...")
            
            # Convert to Python dict/list structure
            python_obj = r_to_python_recursive(r_obj)
            
            # Save as JSON
            csv_path = os.path.expanduser(csv_path)
            output_dir = Path(csv_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving to JSON: {csv_path}")
            with open(csv_path, 'w') as f:
                json.dump(python_obj, f, indent=2, default=str)
            
            print("✓ Conversion to JSON complete")
            print(f"  File size: {Path(csv_path).stat().st_size / 1024:.1f} KB")
            
            # Show structure preview
            if isinstance(python_obj, dict):
                # Filter out metadata keys
                real_keys = [k for k in python_obj.keys() if not k.startswith('_')]
                print(f"\nTop-level components (actual R components): {real_keys}")
                if '_r_class' in python_obj:
                    print(f"R object class: {python_obj['_r_class']}")
            
            return  # Exit early for JSON
            
        else:
            print(f"WARNING: Unexpected R class '{r_class}'.")
            print("Attempting generic conversion or use --inspect to see structure...")
            
            with localconverter(ro.default_converter + pandas2ri.converter):
                result = ro.conversion.rpy2py(r_obj)
            
            if isinstance(result, pd.DataFrame):
                df = result
            elif isinstance(result, np.ndarray):
                # Convert numpy array to DataFrame
                df = pd.DataFrame(result)
                print("Converted numpy array to DataFrame (without row/column names)")
            else:
                # Last resort: try to convert to JSON
                print(f"Cannot convert to CSV. Converting to JSON instead...")
                python_obj = r_to_python_recursive(r_obj)
                
                # Force output to JSON
                csv_path = str(csv_path).rsplit('.', 1)[0] + '.json'
                csv_path = os.path.expanduser(csv_path)
                output_dir = Path(csv_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(csv_path, 'w') as f:
                    json.dump(python_obj, f, indent=2, default=str)
                
                print(f"✓ Saved as JSON: {csv_path}")
                return
        
        # Verify we have a DataFrame for CSV output
        if df is None:
            raise ValueError("Conversion resulted in None")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(df)}. "
                f"Consider using --output file.json or --inspect to see structure."
            )
        
        # Expand user path for output if needed
        csv_path = os.path.expanduser(csv_path)
        
        # Create output directory if it doesn't exist
        output_dir = Path(csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        print(f"Saving to CSV: {csv_path}")
        df.to_csv(csv_path)
        
        print("✓ Conversion complete")
        print(f"  DataFrame dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Display first few rows for verification
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Show column info
        if len(df.columns) > 0:
            print(f"\nColumns: {list(df.columns[:10])}" + 
                  (f" ... (and {len(df.columns) - 10} more)" if len(df.columns) > 10 else ""))
        
        # Show index info
        if df.index is not None and len(df.index) > 0:
            print(f"Row names (index): {list(df.index[:10])}" + 
                  (f" ... (and {len(df.index) - 10} more)" if len(df.index) > 10 else ""))
        
    except ImportError as ie:
        print(
            "ERROR: rpy2 not installed.\n"
            "Install with: pip install rpy2\n"
            "Also ensure R is installed on your system."
        )
        print(f"Import error details: {ie}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert ImmunoStates reference matrix from R to CSV/JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from GitHub (default):
  %(prog)s

  # Use a local .rda file:
  %(prog)s --input /path/to/file.rda

  # Use local file with custom output:
  %(prog)s --input ~/data/immunoStatesMatrix.RData --output matrix.csv

  # Inspect a complex R object to see its structure:
  %(prog)s --input ~/data/tinyMetaObject.RData --inspect

  # Extract a specific component from a complex object:
  %(prog)s --input ~/data/tinyMetaObject.RData --component originalData

  # Save complex object as JSON:
  %(prog)s --input ~/data/tinyMetaObject.RData --output data.json
  
Note: Component names should NOT include $ prefix (e.g., use 'originalData' not '$originalData')
      _r_class is metadata added during conversion and cannot be extracted separately
        """
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to local .rda or .RData file (if not specified, downloads from GitHub)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV or JSON file path (default: ../immunoStates_reference_matrix.csv)"
    )
    parser.add_argument(
        "--variable",
        "-v",
        default=None,
        help="Name of R variable/object to extract from .rda file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--component",
        "-c",
        default=None,
        help="Component to extract from a list/S3 object (e.g., 'originalData', NOT '$originalData')"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect the R object structure without converting (useful for complex objects)"
    )
    parser.add_argument(
        "--keep-rda",
        action="store_true",
        help="Keep the downloaded .rda file (default: delete after conversion)"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None and not args.inspect:
        # Default to data directory
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent
        output_csv = data_dir / "immunoStates_reference_matrix.csv"
    elif args.output:
        output_csv = Path(args.output)
    else:
        output_csv = None
    
    print("="*70)
    print("ImmunoStates Reference Matrix Conversion")
    print("="*70)
    
    # Determine input source
    if args.input:
        # Use local file
        input_path = os.path.expanduser(args.input)
        if not os.path.exists(input_path):
            print(f"ERROR: Input file not found: {input_path}")
            sys.exit(1)
        
        print(f"Input source: Local file")
        print(f"Input file: {input_path}")
        
        # Check file extension
        file_ext = Path(input_path).suffix.lower()
        if file_ext not in ['.rda', '.rdata']:
            print(f"WARNING: File extension '{file_ext}' is unusual for R data files.")
            print(f"Expected .rda or .RData")
            if not args.inspect:
                response = input("Continue anyway? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted.")
                    sys.exit(0)
        
        rda_path = input_path
        should_cleanup = False  # Don't delete local input files
        
    else:
        # Download from GitHub
        print(f"Input source: GitHub download")
        rda_path = download_immunostates_rda()
        should_cleanup = not args.keep_rda
    
    # Convert to CSV/JSON or inspect
    try:
        convert_rda_to_csv(rda_path, str(output_csv) if output_csv else "temp.csv", 
                          args.variable, args.component, args.inspect)
        
        if not args.inspect:
            print("\n" + "="*70)
            print("SUCCESS!")
            print("="*70)
            print(f"Output saved to: {output_csv}")
            if output_csv.exists():
                print(f"File size: {Path(output_csv).stat().st_size / 1024:.1f} KB")
            print("\nYou can now use it in your analyses:")
            if output_csv.suffix == '.json':
                print("  import json")
                print(f"  with open('{output_csv}') as f:")
                print("      data = json.load(f)")
            else:
                print("  import pandas as pd")
                print(f"  df = pd.read_csv('{output_csv}', index_col=0)")
        
    finally:
        # Clean up downloaded .rda file if needed
        if should_cleanup and Path(rda_path).exists():
            Path(rda_path).unlink()
            print(f"\nCleaned up temporary file: {rda_path}")


if __name__ == "__main__":
    main()
