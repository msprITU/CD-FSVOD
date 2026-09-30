import json
import os 
import sys
import copy
import glob


seq = sys.argv[1]
part_size = int(sys.argv[2])
json_path = "/root/BHRL/vot_annotation/{}/vot_{}_test.json". format(seq, seq)

save_path = "/root/BHRL/vot_annotation/{}". format(seq)

if not os.path.exists(save_path):
    os.makedirs(save_path)

def split_seq():
    json_file = open(json_path, "r")
    voc_data = json.load(json_file)

    length = len(voc_data["annotations"])

    num_parts = int(length / part_size)

    # Use list slicing to create four parts
    input_list = voc_data["annotations"]
    parts_ann = []
    for i in range(num_parts):
        if i == (num_parts-1):
            parts_ann.append(input_list[i * part_size:])
        else:
            parts_ann.append(input_list[i * part_size: (i + 1) * part_size])

    input_list = voc_data["images"]
    parts_imgs = []
    for i in range(num_parts):
        if i == (num_parts-1):
            parts_imgs.append(input_list[i * part_size:])
        else:
            parts_imgs.append(input_list[i * part_size: (i + 1) * part_size])


    # remove the parts of earlier runs so that only the parts written below are counted
    for old_part in glob.glob(os.path.join(save_path, "{}_part_*.json". format(seq))):
        os.remove(old_part)

    for idx, each_part in enumerate(parts_ann):
        voc_copy = copy.deepcopy(voc_data)
        voc_copy["images"] = parts_imgs[idx]
        voc_copy["annotations"] = each_part

        json_object = json.dumps(voc_copy,  indent=4)
        out_path = os.path.join(save_path, "{}_part_{}.json". format(seq, str(idx)))
        with open(out_path, "w") as outfile:
            outfile.write(json_object)

    return num_parts - 1


def read_json():
    json_path = "/root/BHRL/vot_annotation/person19/person19_part_9.json"
    json_file = open(json_path, "r")
    voc_data = json.load(json_file)
    print(len(voc_data["annotations"]))
    print(len(voc_data["images"]))
    
number = split_seq()
print(number)