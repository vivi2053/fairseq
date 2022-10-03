import faiss
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle as pkl


class KNN_Dstore:
    def __init__(self, vocab_size: int, hidden_size: int = 2048, lamda: int = 0, generation_cfgs=None):
        self.vocab_size = vocab_size
        self.k = generation_cfgs.k
        self.hidden_size = hidden_size
        self.lamda = lamda
        self.key_dtype = np.float32
        self.val_dtype = np.int64
        self.keys = None
        self.vals = None
        self.dstore_combined = generation_cfgs.dstore_combined
        self.faiss_index = self.setup_index(generation_cfgs)

    def concatenate_subsets(self, gen_cfgs, load_keys=True):
        def read_subset_sizes():
            with open(gen_cfgs.dstore_path + "_file_counts.pkl", "rb") as readfile:
                return pkl.load(readfile)

        info_type = "keys" if load_keys else "vals"
        data_type = np.float32 if load_keys else np.int64
        dim = self.hidden_size if load_keys else 1
        subset_sizes = read_subset_sizes()
        if self.dstore_combined:
            full_size = sum([vals for vals in subset_sizes.values()])
            full_array = np.memmap(gen_cfgs.dstore_path+f"_{info_type}_all.npy",
                                   dtype=data_type, mode='r', shape=(full_size, dim))
        else:
            full_size = sum([vals for vals in subset_sizes.values()])
            full_array = np.memmap(gen_cfgs.dstore_path+f"_{info_type}_all.npy",
                                   dtype=data_type, mode='w+', shape=(full_size, dim))

            cur_idx = 0
            for fname, size in subset_sizes.items():
                sub_array = np.memmap(gen_cfgs.dstore_path+f"_{info_type}_{str(fname)}.npy",
                                      dtype=data_type, mode='r', shape=(size, dim))
                full_array[cur_idx:cur_idx+size, :] = sub_array
                cur_idx += size
        return full_array

    def setup_index(self, generation_cfgs):
        if generation_cfgs.faiss_index_file == "path_to_index_file":
            raise ValueError("No index file specified for KNNMT")

        timer_start = time.time()
        faiss_index = faiss.read_index(generation_cfgs.faiss_index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        timer_end = time.time()
        print(f"Reading Faiss index took {timer_end - timer_start:.3f} secs")
        if generation_cfgs.faiss_index_to_gpu:
            print("Moving the Faiss index to GPU...")
            index_ivf = faiss.extract_index_ivf(faiss_index)
            quantizer = index_ivf.quantizer
            gpu_quantizer = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
            index_ivf.quantizer = gpu_quantizer

        faiss_index.nprobe = generation_cfgs.faiss_nprobe

        keys_from_memmap = self.concatenate_subsets(generation_cfgs)
        vals_from_memmap = self.concatenate_subsets(generation_cfgs, load_keys=False)

        if generation_cfgs.load_vals_to_mem:
            print("Loading values to memory")
            timer_start = time.time()
            self.keys = np.zeros((vals_from_memmap.shape[0], self.hidden_size), dtype=self.key_dtype)
            self.vals = np.zeros((vals_from_memmap.shape[0], 1), dtype=self.val_dtype)
            self.keys = keys_from_memmap[:]
            self.vals = vals_from_memmap[:]
            timer_end = time.time()
            print(f"Loading values to memory took {timer_end - timer_start:.3f} secs")

        else:
            self.keys = keys_from_memmap
            self.vals = vals_from_memmap
        return faiss_index

    def get_neighbors(self, queries):
        Q = queries.detach().cpu().numpy()
        # print(Q[:5, :2])
        D, I = self.faiss_index.search(Q, self.k)
        # print(D, I)
        return D, I

    def get_distances(self, queries, neighbors):
        bsz, dim = queries.shape
        queries_vecs = queries.view(bsz, 1, dim).repeat(1, self.k, 1)
        neighbor_vecs = self.keys[neighbors]
        neighbor_vecs = torch.from_numpy(neighbor_vecs).cuda()
        new_distances = torch.sum((queries_vecs - neighbor_vecs) ** 2, dim=2)
        return new_distances

    def get_scores_per_step(self, step_num, queries, pad_idx, knn_temp):
        queries = queries.squeeze(dim=0)
        qnum, qdim = queries.shape[0], queries.shape[1]

        # compute L2 distances with the neighbors
        distances, neighbors = self.get_neighbors(queries)
        # if step_num == 0 or step_num == 1:
        #     print(neighbors[5])
        distances = self.get_distances(queries, neighbors)
        normalized_distances = F.log_softmax(((-1*distances)/knn_temp), dim=1)

        indices = torch.from_numpy(self.vals[neighbors]).long().cuda()
        unique_indices, mappings = torch.unique(indices, return_inverse=True)

        normalized_distances = normalized_distances.unsqueeze(2)

        knn_scores_by_index = torch.full((indices.shape[0], indices.shape[1],
                                         len(unique_indices)), -10000, dtype=torch.float32).cuda()
        knn_vals_by_index = torch.full((indices.shape[0], indices.shape[1],
                                        len(unique_indices)), pad_idx, dtype=torch.long).cuda()

        knn_scores_by_index.scatter_(dim=2, index=mappings, src=normalized_distances)
        knn_vals_by_index.scatter_(dim=2, index=mappings, src=indices)
        knn_scores_by_index = knn_scores_by_index.logsumexp(dim=1)
        knn_vals_by_index = knn_vals_by_index.max(dim=1)[0]

        full_knn_scores = torch.full((qnum, self.vocab_size), -10000, dtype=torch.float32).cuda()
        full_knn_scores.scatter_(dim=1, index=knn_vals_by_index, src=knn_scores_by_index)

        return full_knn_scores
