import os
import json
from PIL import Image

def create_coco_json(dataset_folder):
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id = 0
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(dataset_folder, filename)
            image = Image.open(image_path)
            width, height = image.size

            image_info = {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            }
            coco_data["images"].append(image_info)
            
            image_id += 1

    # Save the JSON file
    json_file_path = "custom_dataset_coco.json"
    with open(json_file_path, "w") as json_file:
        json.dump(coco_data, json_file)

    print("COCO JSON file created successfully!")

# Provide the path to your custom dataset folder
dataset_folder = "/home/abanerjee/Downloads/testset_mappedv2/testset_mapped"
create_coco_json(dataset_folder)
