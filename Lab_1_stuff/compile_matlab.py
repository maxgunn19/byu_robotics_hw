import scipy.io
import numpy as np
import glob
import os

def consolidate_mat_to_txt(file_pattern, output_filename):
    """
    Finds all .mat files matching a pattern, extracts their data,
    and consolidates it into a single text file.
    
    Args:
        file_pattern (str): A glob pattern to find .mat files (e.g., "part3_trial*.mat").
        output_filename (str): The name of the text file to create (e.g., "consolidated_data.txt").
    """
    
    # Find all files matching the pattern and sort them to ensure order (00, 01, 02...)
    mat_files = sorted(glob.glob(file_pattern))
    
    if not mat_files:
        print(f"Error: No files found matching the pattern '{file_pattern}'.")
        print("Please make sure the script is in the same directory as your .mat files.")
        return

    print(f"Found {len(mat_files)} files. Starting consolidation...")

    try:
        # Open the output text file in 'write' mode ('w')
        # This will create the file or overwrite it if it already exists.
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            
            # Loop through each file found
            for mat_file in mat_files:
                filename = os.path.basename(mat_file)
                print(f"Processing: {filename}...")
                
                # Write a header for this file's data in the text file
                f_out.write(f"--- START OF DATA FROM: {filename} ---\n")
                f_out.write("=" * 60 + "\n")
                
                try:
                    # Load the data from the .mat file
                    # This returns a dictionary where keys are variable names
                    data = scipy.io.loadmat(mat_file)
                    
                    # Iterate through all the variables stored in the .mat file
                    for key, value in data.items():
                        # Skip internal metadata keys (which start with '__')
                        if key.startswith('__'):
                            continue
                        
                        f_out.write(f"Variable Name: '{key}'\n")
                        
                        # Check if the data is a numpy array (most .mat data is)
                        if isinstance(value, np.ndarray):
                            f_out.write(f"  Type: NumPy Array\n")
                            f_out.write(f"  Shape: {value.shape}\n")
                            f_out.write(f"  Data:\n")
                            # Use numpy's array2string for a clean, readable format
                            # This prevents huge arrays from printing on one line
                            data_str = np.array2string(value, 
                                                       precision=5, 
                                                       suppress_small=True, 
                                                       max_line_width=120)
                            f_out.write(data_str + "\n")
                            
                        else:
                            # Handle other simple data types (strings, numbers, etc.)
                            f_out.write(f"  Type: {type(value)}\n")
                            f_out.write(f"  Value: {value}\n")
                        
                        f_out.write("-" * 40 + "\n")
                
                except Exception as e:
                    print(f"  ! Error reading {filename}: {e}")
                    f_out.write(f"!!! ERROR processing file {filename}: {e} !!!\n")
                
                # Write a footer for this file's data
                f_out.write(f"--- END OF DATA FROM: {filename} ---\n\n\n")

        print(f"\nSuccess! All data consolidated into '{output_filename}'.")

    except IOError as e:
        print(f"Error: Could not write to output file '{output_filename}'. {e}")
    except ImportError:
        print("Error: The 'scipy' or 'numpy' library is not installed.")
        print("Please install them first by running:")
        print("pip install scipy numpy")

# --- Main execution ---
if __name__ == "__main__":
    # Define the pattern for your trial files
    FILE_PATTERN = "part3_trial*.mat"
    
    # Define the name of the final output file
    OUTPUT_FILE = "consolidated_trial_data.txt"
    
    consolidate_mat_to_txt(FILE_PATTERN, OUTPUT_FILE)
