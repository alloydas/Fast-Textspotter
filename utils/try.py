import json
with open('output.json', 'r') as f:
    data = json.load(f)
print(type(data))
exist = [int(data[i]["image_id"]) for i in range(len(data))]
#print(exist)
not_exist = []
for i in range(8773):
  if i not in exist:
    not_exist.append(i)
print(not_exist)
converted_data = []
for i in not_exist:
    converted_item = {
            "image_id": str(i),
            "text": []
    }
    converted_data.append(converted_item)

with open("output2.json", 'w') as f:
    json.dump(converted_data, f, indent=4)