import scipy.io as sio
import os
import numpy as np  # Import numpy
import sys          # Import sys

"""
This script converts a .mat (MATLAB) file into a .txt file,
dumping the names, types, and string representations of the
variables stored within it.

To use it, set the `mat_file` and `txt_file` variables
in the `main()` function at the bottom of the script,
and then run:
    python mat_to_txt_converter.py
"""

def convert_mat_to_txt(mat_file_path, txt_file_path):
    """
    Reads a .mat file and writes its contents to a .txt file.
    """
    try:
        # Load the .mat file
        # loadmat returns a dictionary
        data = sio.loadmat(mat_file_path)
        print(f"Successfully loaded '{mat_file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{mat_file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    # Set numpy print options to prevent truncation
    # This forces it to print the entire array
    np.set_printoptions(threshold=np.inf)

    try:
        # Open the output .txt file for writing
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(f"--- Contents of {os.path.basename(mat_file_path)} ---\n")
            f.write("=" * 40 + "\n\n")

            # Iterate through all items in the loaded data dictionary
            for key, value in data.items():
                # Skip internal/header keys
                if key.startswith('__'):
                    continue
                
                # Write variable name
                f.write(f"Variable: {key}\n")
                
                # Write variable type
                f.write(f"Type:     {type(value)}\n")
                
                # Write string representation of the variable's value
                # Now this will be the full, non-truncated array
                f.write("Value:\n")
                f.write(f"{str(value)}\n")
                
                # Add a separator for clarity
                f.write("\n" + "-" * 30 + "\n\n")

        print(f"Successfully converted data to '{txt_file_path}'.")

    except IOError:
        print(f"Error: Could not write to file '{txt_file_path}'. Check permissions.")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")

def main():
    """
    Main function to define file paths and run the converter.
    """
    # --- DEFINE YOUR FILE NAMES HERE ---
    # Make sure the .mat file is in the same folder as this script
    mat_file = "part3_trial09.mat"  # <-- SET YOUR INPUT .mat FILE HERE
    txt_file = "output_3_10.txt"     # <-- SET YOUR OUTPUT .txt FILE HERE
    # -------------------------------------

    # Check if input file exists
    if not os.path.exists(mat_file):
        print(f"Error: Input file not found: '{mat_file}'")
        sys.exit(1)

    convert_mat_to_txt(mat_file, txt_file)

if __name__ == "__main__":
    main()

