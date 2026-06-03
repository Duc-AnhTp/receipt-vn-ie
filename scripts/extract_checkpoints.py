import zipfile
import os
import shutil

def extract_zip(zip_path, target_dir):
    print(f"Extracting {zip_path}...")
    if not os.path.exists(zip_path):
        print(f"Error: File not found {zip_path}")
        return False
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # member has format: receipt-vn-ie/checkpoints/donut/receipt_ie/finetune/best_model/...
            if member.startswith("receipt-vn-ie/"):
                # Create relative path by stripping receipt-vn-ie/
                relative_path = member.replace("receipt-vn-ie/", "", 1)
            else:
                relative_path = member
                
            if not relative_path:
                continue
                
            target_path = os.path.join(target_dir, relative_path)
            
            # Create directory if it doesn't exist
            if member.endswith('/'):
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # Read zip entry and write to target_path
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
    print(f"Successfully extracted {zip_path}!")
    return True

if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Extract donut_best.zip
    extract_zip("donut_best.zip", current_dir)
    
    # Extract layoutxlm_best.zip
    extract_zip("layoutxlm_best.zip", current_dir)
    
    print("=== EXTRACT CHECKPOINTS COMPLETED ===")
