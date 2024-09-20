#include <memory>
#include <map>
#include <iostream>
#include <unordered_map>

#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>
#include <derecho/openssl/hash.hpp>

#include "grouped_embeddings_for_search.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-2200-21ac-1755-0002ac220000"
#define MY_DESC     "UDL search within the clusters to find the top K embeddings that the queries close to."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class ClustersSearchOCDPO: public DefaultOffCriticalDataPathObserver {

    // maps from cluster ID -> embeddings of that cluster, 
    // because there could be more than 1 clusters hashed to one node by affinity set.
    std::unordered_map<int, std::unique_ptr<GroupedEmbeddingsForSearch>> clusters_embs;

    // faiss example uses 64. These two values could be set by config in dfgs.json.tmp file
    int emb_dim = 64; // dimension of each embedding
    uint32_t top_k = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search

    int my_id; // the node id of this node; logging purpose


    /***
     * Format the new_keys for the search results of the queries
     * it is formated as client{client_id}qb{querybatch_id}qc{client_batch_query_count}_cluster{cluster_id}_qid{hash(query)}
     * @param new_keys the new keys to be constructed
     * @param key_string the key string of the object
     * @param query_list a list of query strings
    ***/
    inline void construct_new_keys(std::vector<std::string>& new_keys,
                                                       const std::string& key_string, 
                                                       const std::vector<std::string>& query_list) {
        for (const auto& query : query_list) {
            std::string hashed_query;
            try {
                /*** TODO: do we need 32 bytes of hashed key? will simply int be sufficient? */
                uint8_t digest[32];
                openssl::Hasher sha256(openssl::DigestAlgorithm::SHA256);
                const char* query_cstr = query.c_str();
                sha256.hash_bytes(query_cstr, strlen(query_cstr), digest);
                std::ostringstream oss;
                for (int i = 0; i < 32; ++i) {
                    // Output each byte as a decimal value (0-255) without any leading zeros
                    oss << std::dec << static_cast<int>(digest[i]);
                }
                hashed_query = oss.str();
            } catch(openssl::openssl_error& ex) {
                dbg_default_error("Unable to compute SHA256 of typename. string = {}, exception name = {}", query, ex.what());
                throw;
            }
            std::string new_key = key_string + "_qid" + hashed_query;
            new_keys.push_back(new_key);
        }
    }



    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {
        /*** Note: this object_pool_pathname is trigger pathname prefix: /rag/emb/clusteres_search instead of /rag/emb, i.e. the objp name***/
        dbg_default_trace("[Clusters search ocdpo]: I({}) received an object from sender:{} with key={}", worker_id, sender, key_string);
        // 0. get the cluster ID
        int cluster_id;
        std::string cluster_delimiter = "_cluster";  // move this to rag_utils as macro
        bool extracted_clusterid = parse_number(key_string, cluster_delimiter, cluster_id); // TODO: get the cluster ID from the object
        if (!extracted_clusterid) {
            std::cerr << "Error: cluster ID not found in the key_string" << std::endl;
            dbg_default_error("Failed to find cluster ID from key: {}, at clusters_search_udl.", key_string);
            return;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        int client_id = -1;
        int query_batch_id = -1;
        bool usable_logging_key = parse_batch_id(key_string, client_id, query_batch_id); // Logging purpose
        if (!usable_logging_key)
            dbg_default_error("Failed to parse client_id and query_batch_id from key: {}, unable to track correctly.", key_string);
        TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_START,client_id,query_batch_id,cluster_id);
#endif
        // 1. compute knn for the corresponding cluster on this node
        // 1.1. check if local cache contains the embeddings of the cluster
        auto it = this->clusters_embs.find(cluster_id);
        if (it == this->clusters_embs.end()){
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_START,this->my_id,cluster_id,0);
#endif
            this->clusters_embs[cluster_id]= std::make_unique<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim);
            std::string cluster_prefix = "/rag/emb/cluster" + std::to_string(cluster_id);
            int filled_cluster_embs = this->clusters_embs[cluster_id]->retrieve_grouped_embeddings(cluster_prefix,typed_ctxt);
            if (filled_cluster_embs == -1) {
                std::cerr << "Error: failed to fill the cluster embeddings in cache" << std::endl;
                dbg_default_error("Failed to fill the cluster embeddings in cache, at clusters_search_udl.");
                return;
            }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_END,this->my_id,cluster_id,0);
#endif
            it = this->clusters_embs.find(cluster_id);
        }
        // 1.2. get the embeddings of the cluster
        auto& embs = it->second;

        // 2. get the query embeddings from the object

        float* data;
        uint32_t nq;
        std::vector<std::string> query_list;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_DESERIALIZE_START,client_id,query_batch_id,cluster_id);
#endif
        try{
            deserialize_embeddings_and_quries_from_bytes(object.blob.bytes,object.blob.size,nq,this->emb_dim,data,query_list);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to deserialize the query embeddings and query texts from the object." << std::endl;
            dbg_default_error("{}, Failed to deserialize the query embeddings and query texts from the object.", __func__);
            return;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_DESERIALIZE_END,client_id,query_batch_id,cluster_id);
#endif

        // 3. search the top K embeddings that are close to the query
        long* I = new long[this->top_k * nq];
        float* D = new float[this->top_k * nq];
        embs->search(nq,data,this->top_k,D,I);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_FAISS_SEARCH_END,client_id,query_batch_id,cluster_id);
#endif
        dbg_default_trace("[Cluster search ocdpo] Finished knn search for key: {}.", key_string);
        
    std::cout << "Array I (long):" << std::endl;
    for (size_t i = 0; i < this->top_k * nq; ++i) {
        std::cout << I[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Array D (float):" << std::endl;
    for (size_t i = 0; i < this->top_k * nq; ++i) {
        std::cout << D[i] << " ";
    }
    std::cout << std::endl;

        // 4. emit the results to the subsequent UDL query-by-query
        // 4.1 construct new keys for all queries in this search
        std::vector<std::string> new_keys;
        construct_new_keys(new_keys, key_string, query_list);
        dbg_default_trace("[Cluster search ocdpo] constructed new keys: {}.", key_string);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_CONSTRUCT_KEYS_END,client_id,query_batch_id,cluster_id);
#endif
        int idx = 0;
        for (auto it = query_list.begin(); it != query_list.end(); ++it) {
            // 4.2 format the search result
            std::string query_emit_content = serialize_cluster_search_result(this->top_k, I, D, idx, *it);
            Blob blob(reinterpret_cast<const uint8_t*>(query_emit_content.c_str()), query_emit_content.size());
            // 4.3 emit the result
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_EMIT_START,client_id,query_batch_id,cluster_id);
#endif
            emit(new_keys[idx], EMIT_NO_VERSION_AND_TIMESTAMP , blob);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_EMIT_END,client_id,query_batch_id,cluster_id);
#endif
            dbg_default_trace("[Cluster search ocdpo]: Emitted key:{} " ,new_keys[idx]);
            idx ++;
        }
        delete[] I;
        delete[] D;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_END,client_id,query_batch_id,cluster_id);
#endif
        dbg_default_trace("[Cluster search ocdpo]: FINISHED knn search for key: {}.", key_string );
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ClustersSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
        this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
        try{
            if (config.contains("emb_dim")) {
                this->emb_dim = config["emb_dim"].get<int>();
            }
            if (config.contains("top_k")) {
                this->top_k = config["top_k"].get<uint32_t>();
            }
            if (config.contains("faiss_search_type")) {
                this->faiss_search_type = config["faiss_search_type"].get<int>();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to convert emb_dim or top_k from config" << std::endl;
            dbg_default_error("Failed to convert emb_dim or top_k from config, at clusters_search_udl.");
        }
    }
};

std::shared_ptr<OffCriticalDataPathObserver> ClustersSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    ClustersSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext* ctxt,const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<ClustersSearchOCDPO>(ClustersSearchOCDPO::get())->set_config(typed_ctxt,config);
    return ClustersSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho
