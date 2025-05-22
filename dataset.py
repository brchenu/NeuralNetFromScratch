import gzip, urllib.request, struct
from pathlib import Path

# MNIST Google Drive
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]
MAGIC_NUMBER_IMG = 0x00000803
MAGIC_NUMBER_LABEL = 0x00000801

class MNIST():
    def __init__(self): 
        self.test_images = None
        self.test_labels = None
        self.train_images = None
        self.train_labels = None
        self.dataset_path = None
    
    def _download_mnist(self):
        self.dataset_path = Path("dataset")
        self.dataset_path.mkdir(exist_ok=True)

        for file in FILES:
            out_path = self.dataset_path / file.replace('.gz', '')

            if out_path.exists():
                print(f"Already downloaded: {out_path.name}")
                continue

            print(BASE_URL + file)
            with urllib.request.urlopen(BASE_URL + file) as r:
                with gzip.GzipFile(fileobj=r) as uncompressed:
                    with open(out_path, "wb") as out:
                        out.write(uncompressed.read())

        print(f"MNIST dataset downloaded !")

    @staticmethod
    def _parse_images(filename):
        with open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            assert magic == MAGIC_NUMBER_IMG, f"Invalid images magic number"

            image_size = rows * cols
            images = f.read(num_images * image_size)
            images = list(images)

            return [images[i:i+image_size] for i in range(0, len(images), image_size)]
        
    @staticmethod
    def _parse_labels(filename):
        with open(filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))

            assert magic == MAGIC_NUMBER_LABEL, f"Invalid labels magic number"
            labels = f.read(num_labels)
            return list(labels)
        
    @property
    def train_data(self):
        return self.train_images, self.train_labels
    
    @property
    def test_data(self):
        return self.test_images, self.test_labels

    def get_train_subset(self, start, end):
        return self.train_images[start:end], self.train_labels[start:end]

    def get_test_subset(self, start, end):
        return self.test_images[start:end], self.test_labels[start:end]

    @staticmethod
    def print_normalized_image(image):
        density_charset = "@%#*+=-:. "
        line = ""
        for i, pixel in enumerate(image):
            idx = int((len(density_charset) - 1) * pixel)
            line += density_charset[idx] 

            if (i + 1) % 28 == 0:
                print(line)
                line = ""
    
    def load(self):
        self._download_mnist()
        self.train_images = MNIST._parse_images(self.dataset_path / FILES[0].replace('.gz', ''))
        self.train_labels = MNIST._parse_labels(self.dataset_path / FILES[1].replace('.gz', ''))
        self.test_images = MNIST._parse_images(self.dataset_path / FILES[2].replace('.gz', ''))
        self.test_labels = MNIST._parse_labels(self.dataset_path / FILES[3].replace('.gz', ''))

        # Normalize images
        self.train_images = [[pixel / 255.0 for pixel in img] for img in self.train_images]
        self.test_images = [[pixel / 255.0 for pixel in img] for img in self.test_images]

        # Convert labels to one hot encoding vector
        nb_cls = 10
        self.train_labels = [[1.0 if i == label else 0.0 for i in range(nb_cls)] for label in self.train_labels]
        self.test_labels = [[1.0 if i == label else 0.0 for i in range(nb_cls)] for label in self.test_labels]

        print(f"MNIST dataset ready !")