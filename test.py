# IDX file format: https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html

def printBytes(data: bytes):
    print(' '.join("{:08b}".format(b) for b in magic_number))

# IDX file parsing func

if __name__ == "__main__":
    with open("input/train-images.idx3-ubyte", "rb") as f:
        data = f.read()

    magic_number = data[:4]
    printBytes(f"magic number: {magic_number}")
    
    data_type = data[2] # 00001000 => 8 => Mean Unsigned byte
    dimension = data[3]
