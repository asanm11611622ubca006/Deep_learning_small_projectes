import os
import glob

def reconstruct_file(filepath):
    # Find all parts for the file
    parts = sorted(glob.glob(f"{filepath}.part*"), key=lambda x: int(x.split(".part")[-1]))
    if not parts:
        print(f"No parts found for {filepath}")
        return

    print(f"Reconstructing {filepath} from {len(parts)} parts...")
    with open(filepath, 'wb') as output_file:
        for part in parts:
            print(f"Adding {part}...")
            with open(part, 'rb') as f:
                output_file.write(f.read())
    
    print(f"Reconstruction complete: {filepath}")

if __name__ == "__main__":
    target_file = r"Image classification using_CNN\model.h5"
    reconstruct_file(target_file)
    # Optional: Delete parts after reconstruction
    # for part in glob.glob(f"{target_file}.part*"):
    #     os.remove(part)
