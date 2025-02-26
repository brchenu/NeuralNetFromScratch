from nn import MLP
from mnist_parser import parse_images, parse_labels

images = parse_images("dataset/training/train-images.idx3-ubyte")
labels = parse_labels("dataset/training/train-labels.idx1-ubyte")

test_images = parse_images("dataset/test/t10k-images.idx3-ubyte")
test_labels = parse_labels("dataset/test/t10k-labels.idx1-ubyte")

def split_in_batch(lst, x): 
    return [lst[i:i + x] for i in range(0, len(lst), x)]

def to_list(label):
    label_lst = [0.0] * 10
    label_lst[label] = 1.0
    return label_lst

batches_img = split_in_batch(images, 300)
batches_label = split_in_batch(labels, 300) 

print(f"batches len: {len(batches_img)}")
print(f"batches len: {len(batches_label)}")

mlp = MLP([[728, 128], [128, 64], [64, 10]])

epoch = 100
learning_rate = 0.1

for images, labels in zip(batches_img, batches_label):
    total_loss = 0
    for img, label in zip(images, labels):
        loss = mlp.train(img, to_list(label), learning_rate)
        total_loss += loss
    
    print(f"Batch loss: {total_loss/len(images)}")