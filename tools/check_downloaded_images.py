import json
import os

# This script should be run on the machine with the AVA dataset downloaded. The
# samples.json file (aka data/AVA/ava_labels_train.json) should also be copied to
# the same machine. This script will output a new version of the training labels
# JSON file which omits any image files that are not present after the images have
# been downloaded and extracted from their .7z zip files
#
# This is useful because even after running downloading and extraction multiple
# times, there are still a few (~20 out of hundreds of thousands) images which
# are specified in the JSON file of labels, but which are not present on the
# filesystem. These files cause training to crash randomly (typically in the first
# 100 training steps. As long as only a few files are missing, removing these files
# does little harm to the balance of classes in the dataset and allows training to
# proceed smoothly.
with open('/home/ubuntu/samples.json', 'r') as f:
    samples_json = json.load(f)
    samples = set([sample["image_id"].encode("ascii") for sample in samples_json])

    downloaded_images = set([
        os.path.splitext(filename)[0]
        for filename in
        os.listdir('/home/ubuntu/Downloads/AVA_dataset')
        if 'jpg' in filename
    ])

    missing = samples.difference(downloaded_images)
    print(missing)
    print(len(missing))

    with open('/home/ubuntu/pruned_samples.json', 'w') as out:
        json.dump([
            sample for sample in samples_json
            if sample["image_id"].encode("ascii") not in missing
        ], out)


