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

class MNIST():
    def __init__(self): 
        self.test_images = None
        self.test_labels = None
        self.images = None
        self.labels = None
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

    def _parse_images(filename):
        with open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            assert magic == 0x00000803, f"Invalid images magic number"  # 0x00000803 = Image file

            image_size = rows * cols
            images = f.read(num_images * image_size)
            images = list(images)

            return [images[i:i+image_size] for i in range(0, len(images), image_size)]
        
    def parse_labels(filename):
        with open(filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))

            assert magic == 0x00000801, f"Invalid labels magic number"  # 0x00000801 = Label file
            labels = f.read(num_labels)
            return list(labels)

    def load(self):
        self._download_mnist()
        self.images = MNIST._parse_images(self.dataset_path / FILES[0].replace('.gz', ''))
        self.labels = MNIST.parse_labels(self.dataset_path / FILES[1].replace('.gz', ''))
        self.test_images = MNIST._parse_images(self.dataset_path / FILES[2].replace('.gz', ''))
        self.test_labels = MNIST.parse_labels(self.dataset_path / FILES[3].replace('.gz', ''))

        print(f"MNIST dataset ready !")