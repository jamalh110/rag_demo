import numpy as np
import faiss
import pickle
import os 

EMBEDDINGS_LOC = './gist'

def fvecs_read(filename, dtype=np.float32, c_contiguous=True):
    fv = np.fromfile(filename, dtype=dtype)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

base = fvecs_read('./gist/gist_base.fvecs')
print("Base shape", base.shape)
#print(base[0])


groundtruth = fvecs_read('./gist/gist_groundtruth.ivecs', np.int32)
print("groundtruth shape",groundtruth.shape)
#print(groundtruth[0])

query = fvecs_read('./gist/gist_query.fvecs')
print("query shape", query.shape)


dimension = base.shape[1]  # Assumes emb_list is a 2D array (num_embeddings, embedding_dim)
print("dimension", dimension)
# Create a FAISS index, here we're using an IndexFlatL2 which is a basic index with L2 distance
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(base)

# Search for the nearest 5 neighbors
k = 5  # number of nearest neighbors
distances, indices = index.search(query[0].reshape(1, -1), k)

# Print the results
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)
print("groundtruth", groundtruth[0])


doc_emb_map = {0:{}}
for i in range(len(base)):
    doc_emb_map[0][i] = i

querytexts = []

for i in range(len(query)):
    querytexts.append("Query " + str(i))


testcentroid = np.zeros((1, 960))


os.makedirs(EMBEDDINGS_LOC, exist_ok=True)

with open(f'{EMBEDDINGS_LOC}/centroids.pkl', 'wb') as file:
    pickle.dump(testcentroid, file)

with open(f'{EMBEDDINGS_LOC}/embeddings_list.pkl', 'wb') as f:
    pickle.dump(base, f)

with open(f'{EMBEDDINGS_LOC}/cluster_0.pkl', 'wb') as f:
    pickle.dump(base, f)

with open(f'{EMBEDDINGS_LOC}/doc_emb_map.pkl', 'wb') as f:
    pickle.dump(doc_emb_map, f)


np.savetxt(f'{EMBEDDINGS_LOC}/groundtruth.csv', groundtruth, delimiter=",", fmt='%i')
np.savetxt(f'{EMBEDDINGS_LOC}/query.csv', querytexts, fmt="%s")
np.savetxt(f'{EMBEDDINGS_LOC}/query_emb.csv', query, delimiter=",")



# np.save("gist/embeddings.npy", base, False)
# np.save("gist/groundtruth.npy", groundtruth, False)
# np.save("gist/query.npy", query, False)

# np.save("gist/cluster_0.npy", base, False)