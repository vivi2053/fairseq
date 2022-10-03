import faiss
import argparse
import numpy as np
import pickle as pkl
import time

# key_dtype = np.float32
# val_dtype = np.int


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dstore-path", type=str, default="/home/vipul/Projects/misc-code-tasks/nlp-111/aspec_dstore/knnmt_dstore",
                        help="path to where all the key and val files are stored")
    parser.add_argument('--index-save-path', type=str, default="/home/ubuntu/filesystem/vipul/Knnmt_Dstores/index.trained",
                        help='path to the file where the index will be saved')
    parser.add_argument("--key-size", type=int, default=2048, help="dimension of each key")
    parser.add_argument("--ncentroids", type=int, default=20000, help="number of centroids that faiss should learn")
    parser.add_argument("--quant-size", type=int, default=256, help="size of the quantized vector")
    parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
    parser.add_argument('--gpu', action='store_true',
                        help="training the index from a mmap that only holds keys needed for training.")
    parser.add_argument("--index-type", type=str, default="ivfpq", help="the type of index to train")
    parser.add_argument("--test", action="store_true", help="train or test index")
    parser.add_argument("--dstore-combined", action="store_true", help="Dstore combined in a single key and value file or not")
    parser.add_argument("--train-vec-num", type=str, default="all",
                        help="number of vectors to use for training the faiss index")
    # parser.add_argument("--", type=str, help="")
    args = parser.parse_args()
    return args


def read_subset_sizes(args):
    with open(args.dstore_path + "_file_counts.pkl", "rb") as readfile:
        return pkl.load(readfile)


def concatenate_subsets(args, subset_sizes, load_keys=True):
    data_size = args.key_size if load_keys else 1
    info_type = "keys" if load_keys else "vals"
    data_type = np.float32 if load_keys else np.int64
    full_size = sum([vals for vals in subset_sizes.values()])
    full_array = np.memmap(args.dstore_path+f"_{info_type}_all.npy", dtype=data_type, mode='w+', shape=(full_size, data_size))

    cur_idx = 0
    for fname, size in subset_sizes.items():
        sub_array = np.memmap(args.dstore_path+f"_{info_type}_{str(fname)}.npy",
                              dtype=data_type, mode='r', shape=(size, data_size))
        full_array[cur_idx:cur_idx+size, :] = sub_array
        cur_idx += size
    return full_array


def train_ivfpq_index(args, full_keys, ngpu):
    quantizer = faiss.IndexFlatL2(args.key_size)
    bits_ = 8  # number of bits in each centroid
    index = faiss.IndexIVFPQ(quantizer, args.key_size, args.ncentroids, args.quant_size, bits_)
    index.nprobe = args.probe  # number of centroids to probe during search?

    if args.gpu:
        print("training ivfpq index on gpu...")
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(args.key_size), ngpu=ngpu)
        index.clustering_index = clustering_index
        if args.train_vec_num == 'all':
            index.train(full_keys[:].astype(np.float32))
        else:
            index.train(full_keys[:int(args.train_vec_num)].astype(np.float32))
        print("ntotal: {0}".format(index.ntotal))
    else:
        print("training ivfpq index on cpu...might be slow")
        if args.train_vec_num == 'all':
            index.train(full_keys[:].astype(np.float32))
        else:
            index.train(full_keys[:int(args.train_vec_num)].astype(np.float32))
    return index


def train_flatl2_index(args):
    print('creating FlatL2 index')
    index = faiss.IndexFlatL2(args.key_size)
    return index


def load_full_data(args, subset_sizes, load_keys=True):
    data_size = args.key_size if load_keys else 1
    info_type = "keys" if load_keys else "vals"
    data_type = np.float32 if load_keys else np.int64
    full_size = sum([v for v in subset_sizes.values()])
    full_array = np.memmap(args.dstore_path+f"_{info_type}_all.npy", dtype=data_type, mode='r', shape=(full_size, data_size))
    return full_array


def train_index(args, subset_sizes, ngpu):
    if args.dstore_combined:
        all_keys = load_full_data(args, subset_sizes)
    else:
        all_keys = concatenate_subsets(args, subset_sizes)

    start_time = time.time()

    # training index
    if args.index_type == "ivfpq":
        index = train_ivfpq_index(args, all_keys, ngpu)
    elif args.index_type == "flatl2":
        index = train_flatl2_index(args)
    else:
        print("Invalid index type")
        exit()

    print(f"Index is trained: {index.is_trained}")
    train_end_time = time.time()
    print(f"Training took {train_end_time-start_time :.3f} secs")

    # Adding keys
    add_start_time = time.time()
    print("Adding keys to index")
    index.add(all_keys[:])
    add_end_time = time.time()
    print("Adding keys took {:.3f} s".format(add_end_time-add_start_time))
    print("ntotal: {0}".format(index.ntotal))

    faiss.write_index(index, args.index_save_path)
    print('Writing index took {:.3f} s'.format(time.time()-add_end_time))


def test_index(args, subset_sizes):
    from random import randint
    # to ensure the index is correctly trained
    # randomly sample some keys from the key files and check if they fall in the top k neighbors
    all_keys = concatenate_subsets(args, subset_sizes)
    all_vals = concatenate_subsets(args, subset_sizes, load_keys=False)
    print(all_vals.shape)
    rand_idx = randint(0, 9999)
    test_key = all_keys[rand_idx:rand_idx+1, :]
    test_val = all_vals[rand_idx, :]
    print("test value: {0}".format(test_val))
    print(test_key.shape, test_val.shape)

    trained_index = read_index(args, all_keys)
    print(trained_index.ntotal)
    D, I = trained_index.search(test_key, 8)
    print(I[0][0], all_vals[I[0][0]])


def read_index(args, keys):
    faiss_index = faiss.read_index(args.index_save_path)
    print("ntotal: {0}".format(faiss_index.ntotal))
    return faiss_index


def main():
    args = parse_arguments()
    ngpu = faiss.get_num_gpus()
    print(f"number of gpus: {ngpu}")

    subset_sizes = read_subset_sizes(args)
    if args.test:
        test_index(args, subset_sizes)
    else:
        train_index(args, subset_sizes, ngpu)


if __name__ == "__main__":
    main()
