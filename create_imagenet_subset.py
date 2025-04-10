from datasets import load_dataset, Dataset, Image, ClassLabel
from tqdm.autonotebook import tqdm

data = load_dataset('imagenet-1k', split='train', trust_remote_code=True)

all_labels = data.features['label'].names
desired_labels = [235, 242, 282, 717, 980]
count_per_label = 100

images = []
labels = []
counts = [0 for _ in desired_labels]

pbar = tqdm(total=count_per_label * len(desired_labels))
for x in tqdm(data):
    if x['label'] in desired_labels:
        i = desired_labels.index(x['label'])
        img = x['image']
        if counts[i] >= count_per_label: continue
        if min(img.size) < 350: continue  # ignore small images
        w, h = img.size
        s = min(w, h)
        new_img = img.crop((w//2-s//2, h//2-s//2, w//2+s//2, h//2+s//2)).resize((512,512))
        images.append(new_img)
        labels.append(i)
        counts[i] += 1
        pbar.update(1)
        if sum(counts) == count_per_label * len(desired_labels):
            break

new_dataset = Dataset.from_dict({'image': images, 'label': labels})
new_dataset = new_dataset.cast_column('image', Image(decode=True, id=None))
new_dataset = new_dataset.cast_column('label', ClassLabel(names=[all_labels[x] for x in desired_labels]))

new_dataset.save_to_disk('imagenet_subset')

# new_dataset.push_to_hub('USERNAME/imagenet_subset', private=True)
