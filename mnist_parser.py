import struct

def parse_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))

        assert magic == 0x00000801, f"Invalid labels magic number"  # 0x00000801 = Label file
        labels = f.read(num_labels)
        return list(labels)

def parse_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 0x00000803, f"Invalid images magic number"  # 0x00000803 = Image file

        image_size = rows * cols
        images = f.read(num_images * image_size)
        images = list(images)
        return [images[i * image_size:(i + 1) * image_size] for i in range(num_images)]