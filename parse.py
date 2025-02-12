import numpy as np

# IDX file format: https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html

# format: 
#   magic number
#   size in dimension 1
#   size in dimension 2
#   ...
#   size in dimension N
#   data

MAGIC_NB_IN_BYTES = 4 # magic number size in bytes
DIM_IN_BYTES = 4 # dimension size in bytes

def printBytes(data: bytes, msg:str=""):
    print(f"{msg} {' '.join("{:08b}".format(b) for b in data)}")

# Images Utils

# ASCII density line

DENSITY_LINE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'."
nb_char = len(DENSITY_LINE)

def printImgToFile(data: bytes):
    with open("output.txt", "w") as f:
        line = ""
        for idx, b in enumerate(data):
            assert(0 <= b <= 255)
            v = 255 - b
            index = min(int((nb_char / 255) * v), nb_char - 1)

            line += DENSITY_LINE[index]
        
            if (idx + 1) % 28 == 0:  # Becareful expect 28x28 images
                f.write(line + "\n")
                line = ""


def normalize_image(image):
    """Normalize pixel values from [0, 255] to [0, 1]"""
    return image / 255.0


# Temporay function to be able to call it from another python script
def parseImages():
    with open("input/train-images.idx3-ubyte", "rb") as f:
        data = f.read()

    magic_number = data[:4]
    printBytes(magic_number, "Magic number:")
    
    data_type = data[2] # 00001000 => 8 => Mean Unsigned byte
    dimension = data[3]

    dim_sizes = data[4 : (DIM_IN_BYTES * dimension) + 4]
    # dims = [dim_sizes[0 + (DIM_IN_BYTES * i): (DIM_IN_BYTES * i) + DIM_IN_BYTES]for i in range(dimension)]

    dim1 = int.from_bytes(dim_sizes[0:4])
    dim2 = int.from_bytes(dim_sizes[4:8])
    dim3 = int.from_bytes(dim_sizes[8:12])

    print(f"dim1: {dim1} / dim2: {dim2} / dim3: {dim3}")

    data_offset = MAGIC_NB_IN_BYTES + dimension * DIM_IN_BYTES

    image_size = dim2 * dim3

    images = []
    for i in range(dim1):
        offset = data_offset + (image_size * i)
        images.append(data[offset : offset + image_size])

    return images

if __name__ == "__main__":
    
    images = parseImages()

    IMG_IDX = 3000

    printImgToFile(images[IMG_IDX])

    arr = np.frombuffer(images[IMG_IDX], dtype=np.uint8).reshape(-1, 1)
    arr = normalize_image(arr)
    print(arr)