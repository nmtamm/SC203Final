import os
import zipfile
import shutil

src_dir = "D:/InverseCookingTemp/Zip_according_to_retrieval"
temp_extract_dir = os.path.join(src_dir, "temp_extract")

for zip_name in os.listdir(src_dir):
    if not zip_name.endswith(".zip"):
        continue
    zip_path = os.path.join(src_dir, zip_name)
    print(f"Processing {zip_name}...")

    # 1. Extract zip to temp directory
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # 2. Find the subfolder (assume only one)
    subfolders = [
        f
        for f in os.listdir(temp_extract_dir)
        if os.path.isdir(os.path.join(temp_extract_dir, f))
    ]
    if not subfolders:
        print(f"No subfolder found in {zip_name}, skipping.")
        continue
    subfolder_path = os.path.join(temp_extract_dir, subfolders[0])

    # 3. Collect all JSON files in the subfolder
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith(".json")]

    # 4. Create a new zip with only the JSON files at the root
    new_zip_path = os.path.join(src_dir, f"{zip_name}")
    with zipfile.ZipFile(new_zip_path, "w", zipfile.ZIP_DEFLATED) as new_zip:
        for json_file in json_files:
            json_path = os.path.join(subfolder_path, json_file)
            new_zip.write(json_path, arcname=json_file)
    print(f"Created {new_zip_path}")

    # 5. Clean up
    shutil.rmtree(temp_extract_dir)
