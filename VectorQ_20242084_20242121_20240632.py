import os
import numpy as np
import json
import math
from PIL import Image


def pad_image(image, block_h, block_w):
    height, width = image.shape

    pad_h = (block_h - height % block_h) % block_h
    pad_w = (block_w - width % block_w) % block_w

    padded_image = np.pad(
        image,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0
    )

    return padded_image


def extract_blocks(image, block_h, block_w):
    h, w = image.shape
    blocks = []

    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = image[i:i+block_h, j:j+block_w]
            vector = block.flatten()
            blocks.append(vector)

    return np.array(blocks)


def find_nearest_codeword(vector, codebook):
    vector = np.asarray(vector, dtype=float)
    codebook = np.asarray(codebook, dtype=float)

    distances = np.sum(np.abs(codebook - vector), axis=1)

    index = np.argmin(distances)

    return int(index)


def build_codebook(vectors, codebook_size):
    vectors = np.array(vectors, dtype=float)
    if codebook_size < 1:
        raise ValueError("Codebook size must be at least 1")

    if codebook_size > len(vectors):
        raise ValueError("Codebook size cannot exceed number of vectors")
    codebook = np.mean(vectors, axis=0).reshape(1, -1)
    while len(codebook) < codebook_size:
        new_codebook = []
        remaining = codebook_size - len(codebook)
        for i, cw in enumerate(codebook):
            if i < remaining:
                new_codebook.append(cw - 1)
                new_codebook.append(cw + 1)
            else:
                new_codebook.append(cw)
        codebook = np.array(new_codebook)

        for _ in range(100):
            idx = assign_all(vectors, codebook)
            old_codebook = codebook.copy()
            for i in range(len(codebook)):
                members = vectors[idx == i]
                if len(members) > 0:
                    codebook[i] = members.mean(axis=0)

            if np.allclose(old_codebook, codebook):
                break
    return codebook

def assign_all(vectors, codebook):
    indices = np.empty(len(vectors), dtype=int)
    for s in range(0, len(vectors), 2048):  # chunks to save memory
        chunk = vectors[s:s+2048]
        d = np.abs(chunk[:, None, :] - codebook[None, :, :]).sum(axis=2)
        indices[s:s+2048] = d.argmin(axis=1)
    return indices


def get_indices(vectors, codebook):
    return assign_all(np.asarray(vectors, dtype=float), codebook).tolist()


def encode_indices(indices, codebook_size):
    bits = max(1, math.ceil(math.log2(codebook_size)))
    bitstream = ""
    for index in indices:
        binary = bin(int(index))[2:]
        binary = binary.zfill(bits)
        bitstream += binary

    return bitstream, bits


def save_codebook(codebook, filename):
    np.save(filename, codebook)


def save_bitstream(bitstream, filename):
    while len(bitstream) % 8 != 0:
        bitstream += "0"

    with open(filename, "wb") as file:
        for i in range(0, len(bitstream), 8):
            byte = bitstream[i:i+8]
            file.write(bytes([int(byte, 2)]))


def save_metadata(metadata, filename):
    with open(filename, "w") as file:
        json.dump(metadata, file, indent=4)


def calculate_compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return 0
    ratio = original_size / compressed_size
    return ratio


def compress_image(image, block_h, block_w, codebook_size):
    original_h, original_w = image.shape
    padded_image = pad_image(image, block_h, block_w)
    padded_h, padded_w = padded_image.shape
    vectors = extract_blocks(padded_image, block_h, block_w)
    codebook = build_codebook(vectors, codebook_size)
    if len(codebook) != codebook_size:
        raise ValueError("Failed to build the requested codebook size")
    indices = get_indices(vectors, codebook)
    metadata = {
        "original_height": original_h,
        "original_width": original_w,
        "padded_height": padded_h,
        "padded_width": padded_w,
        "block_height": block_h,
        "block_width": block_w,
        "codebook_size": len(codebook)
    }

    return codebook, indices, metadata


# Decompression
def load_codebook(filename):
    return np.load(filename)



def load_metadata(filename):
    with open(filename, "r") as file:
        return json.load(file)


def load_bitstream(filename):
    bitstream = ""

    with open(filename, "rb") as file:
        data = file.read()

    for byte in data:
        bitstream += format(byte, "08b")
    return bitstream



def decode_indices(bitstream, bits, num_blocks):
    indices = []

    for k in range(num_blocks):
        chunk = bitstream[k * bits:(k + 1) * bits]
        indices.append(int(chunk, 2))

    return indices


def rebuild_image(indices, codebook, metadata):
    padded_h = metadata["padded_height"]
    padded_w = metadata["padded_width"]
    block_h = metadata["block_height"]
    block_w = metadata["block_width"]
    original_h = metadata["original_height"]
    original_w = metadata["original_width"]

    blocks_per_row = padded_w // block_w

    image = np.zeros((padded_h, padded_w), dtype=float)

    for k, index in enumerate(indices):
        i = (k // blocks_per_row) * block_h
        j = (k % blocks_per_row) * block_w

        block = codebook[index].reshape(block_h, block_w)
        image[i:i+block_h, j:j+block_w] = block


    image = image[:original_h, :original_w]

    image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return image


def decompress_image(codebook, bitstream, metadata):
    bits = metadata["bits_per_index"]
    num_blocks = metadata["num_blocks"]

    indices = decode_indices(bitstream, bits, num_blocks)
    return rebuild_image(indices, codebook, metadata)


def compress_file(image_path, block_h, block_w, codebook_size, out_name="compressed"):
    image = np.array(Image.open(image_path).convert("L"))

    codebook, indices, metadata = compress_image(
        image, block_h, block_w, codebook_size
    )

    bitstream, bits = encode_indices(indices, len(codebook))
    metadata["bits_per_index"] = bits
    metadata["num_blocks"] = len(indices)
    codebook_file = out_name + "_codebook.npy"
    bitstream_file = out_name + "_bitstream.bin"
    metadata_file = out_name + "_metadata.json"
    codebook_to_save = codebook.astype(np.float32)

    save_codebook(codebook_to_save, codebook_file)
    save_bitstream(bitstream, bitstream_file)
    save_metadata(metadata, metadata_file)

    original_size = image.shape[0] * image.shape[1]
    compressed_size = (os.path.getsize(bitstream_file) + os.path.getsize(codebook_file) + os.path.getsize(metadata_file))
    ratio = calculate_compression_ratio(original_size, compressed_size)

    print("Original size   :", original_size, "bytes")
    print("Compressed size :", compressed_size, "bytes (all files)")
    print("Compression ratio:", round(ratio, 3))
    print("Saved:", codebook_file, bitstream_file, metadata_file)
    return ratio


def decompress_file(in_name="compressed", output_path="decompressed.png"):
    codebook = load_codebook(in_name + "_codebook.npy")
    metadata = load_metadata(in_name + "_metadata.json")
    bitstream = load_bitstream(in_name + "_bitstream.bin")

    image = decompress_image(codebook, bitstream, metadata)
    Image.fromarray(image).save(output_path)
    print("Decompressed image saved as:", output_path)
    return image


def main():
    print("Vector Quantization")
    print("1. Compress")
    print("2. Decompress")
    choice = input("Choose 1 or 2: ").strip()
    if choice == "1":
        image_path = input("Image path: ").strip()
        block_h = int(input("Block height: "))
        block_w = int(input("Block width: "))
        codebook_size = int(input("Codebook size: "))
        if block_h <= 0 or block_w <= 0:
            print("Block dimensions must be positive")
            return

        if codebook_size <= 0:
            print("Codebook size must be positive")
            return
        compress_file(image_path, block_h, block_w, codebook_size)

    elif choice == "2":
        output_path = input("Output image name (example: out.png): ").strip()
        decompress_file("compressed", output_path)

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
