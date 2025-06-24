import os

def bundle_project_files(file_paths, output_file="project_bundle.txt"):
    """
    Concatenates the content of multiple files into a single text file.

    Args:
        file_paths (list): A list of paths to the files to be included.
        output_file (str): The name of the output file.
    """
    try:
        with open(output_file, 'w') as outfile:
            print(f"🚀 Starting to bundle {len(file_paths)} files into '{output_file}'...")
            for file_path in file_paths:
                if os.path.exists(file_path):
                    outfile.write(f"\n{'='*40}\n")
                    outfile.write(f"📄 START OF FILE: {file_path}\n")
                    outfile.write(f"{'='*40}\n\n")
                    
                    with open(file_path, 'r') as infile:
                        outfile.write(infile.read())
                    
                    outfile.write(f"\n\n{'='*40}\n")
                    outfile.write(f"📄 END OF FILE: {file_path}\n")
                    outfile.write(f"{'='*40}\n")
                    print(f"  ✅ Successfully added: {file_path}")
                else:
                    print(f"  ⚠️  Warning: File not found and will be skipped: {file_path}")
        
        print(f"\n🎉 Successfully created '{output_file}' with all specified project files.")

    except IOError as e:
        print(f"❌ Error: Could not write to output file '{output_file}'. Reason: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- List of files to be bundled ---
    # You can easily add or remove file paths from this list.
    files_to_bundle = [
        "run.py",
        "graph.py",
        "agents.py",
        "schemas.py",
        "prompts.py",
        "config.yaml",
        "config_utils.py",
        "viz/live_dashboard.py",
        "utils/event_bus.py"
    ]
    
    bundle_project_files(files_to_bundle) 