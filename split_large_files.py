import os

def split_file(filepath, chunk_size=95 * 1024 * 1024):  # 95MB chunks
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    file_size = os.path.getsize(filepath)
    print(f"Splitting {filepath} ({file_size / (1024*1024):.2f} MB)...")

    with open(filepath, 'rb') as f:
        chunk_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_filename = f"{filepath}.part{chunk_num}"
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"Created {chunk_filename}")
            chunk_num += 1

if __name__ == "__main__":
    target_file = r"Image classification using_CNN\model.h5"
    split_file(target_file)
