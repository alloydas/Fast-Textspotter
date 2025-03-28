import json

def convert_json_format(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)



    converted_data = []
    for item in data:
        xmin = 1000000
        ymin = 1000000
        xmax = 0
        ymax = 0
        for i in range(len(item['polys'])):
            pt = item['polys'][i]
            if pt[0]<xmin:
                xmin = pt[0]
            if pt[0]>xmax:
                xmax = pt[0]

            if pt[1]<ymin:
                ymin = pt[1]
            if pt[1]>ymax:
                ymax = pt[1]
        converted_item = {
            "image_id": str(item["image_id"]),
            "text": [
                {
                    "transcription": str(item["rec"]),
                    "confidence": float(item["score"]),
                    "vertices": [
                        [float(xmin), float(ymin)],
                        [float(xmax), float(ymin)],
                        [float(xmax), float(ymax)],
                        [float(xmin), float(ymax)]
                    ]
                }
            ]
        }
        converted_data.append(converted_item)

    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)

# Example usage
input_file = 'synt+mlt+ctw_results.json'
output_file = 'output.json'
convert_json_format(input_file, output_file)