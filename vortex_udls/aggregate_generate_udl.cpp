#include <algorithm>
#include <memory>
#include <map>
#include <iostream>
#include <tuple>
#include <unordered_map>

#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>

#include "rag_utils.hpp"

namespace derecho{
namespace cascade{

#define MY_UUID     "11a3c123-3300-31ac-1866-0003ac330000"
#define MY_DESC     "UDL to aggregate the knn search results for each query from the clusters and run LLM with the query and its top_k closest docs."
#define SKIP_DOC_RETRIEVAL false

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}


struct ClusterSearchResults{
    const std::string query_text;
    int total_cluster_num = 0;
    std::vector<int> collected_cluster_ids;
    bool collected_all_results = false;
    int top_k = 0;
    // min heap to keep the top_k docIndex across clusters' results
    std::priority_queue<DocIndex> agg_top_k_results;

    ClusterSearchResults(const std::string& query_text, int total_cluster_num, int top_k): 
                         query_text(query_text), total_cluster_num(total_cluster_num), top_k(top_k) {}

    bool is_all_results_collected(){
        if(static_cast<int>(collected_cluster_ids.size()) == total_cluster_num){
            collected_all_results = true;
        }
        // print out the docIndex in the min heap for debugging
        std::priority_queue<DocIndex> tmp = agg_top_k_results;
        return collected_all_results;
    }

    bool add_cluster_result(int cluster_id, std::vector<DocIndex> cluster_results){
        if(std::find(collected_cluster_ids.begin(), collected_cluster_ids.end(), cluster_id) != collected_cluster_ids.end()){
            std::cerr << "ERROR: cluster id=" << cluster_id << " search result is already collected for query=" << this->query_text << std::endl;
            dbg_default_error("{} received same cluster={} result for query={}.", __func__, cluster_id, this->query_text);
            return false;
        }
        this->collected_cluster_ids.push_back(cluster_id);
        // Add the cluster_results to the min_heap, and keep the size of the heap to be top_k
        for (const auto& doc_index : cluster_results) {
            if (static_cast<int>(agg_top_k_results.size()) < top_k) {
                agg_top_k_results.push(doc_index);
            } else if (doc_index < agg_top_k_results.top()) {
                agg_top_k_results.pop();
                agg_top_k_results.push(doc_index);
            }
        }
        return true;
    }
};

class AggGenOCDPO: public DefaultOffCriticalDataPathObserver {

    int top_k = 5; // final top K results to use for LLM
    int top_num_centroids = 4; // number of top K clusters need to wait to gather for each query
    int include_llm = false; // 0: not include, 1: include

    std::unordered_map<int, std::unordered_map<long, std::string>> doc_tables; // cluster_id -> emb_index -> pathname
    /*** TODO: use a more efficient way to store the doc_contents cache */
    std::unordered_map<int,std::unordered_map<long, std::string>> doc_contents; // {cluster_id0:{ emb_index0: doc content0, ...}, cluster_id1:{...}, ...}
    /*** query_result: query_text -> ClusterSearchResults 
     *   is a UDL local cache to store the cluster search results for queries that haven't notified the client 
     *   (due to not all cluster results are collected)
    */
    std::unordered_map<std::string, std::unique_ptr<ClusterSearchResults>> query_results; // 


    int my_id; // the node id of this node; logging purpose


    bool load_doc_table(DefaultCascadeContextType* typed_ctxt, int cluster_id){
        if (doc_tables.find(cluster_id) != doc_tables.end()) {
            return true;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_START, my_id, 0, cluster_id);
#endif
        // 0. check the keys for this grouped embedding objects stored in cascade
        //    because of the message size, the map for one cluster may split into multiple chunks stored in Cascade
        bool stable = 1; 
        persistent::version_t version = CURRENT_VERSION;
        std::string table_prefix = "/rag/doc/emb_doc_map/cluster" + std::to_string(cluster_id);
        auto keys_future = typed_ctxt->get_service_client_ref().list_keys(version, stable, table_prefix);
        std::vector<std::string> map_obj_keys = typed_ctxt->get_service_client_ref().wait_list_keys(keys_future);
        if (map_obj_keys.empty()) {
            std::cerr << "Error: " << table_prefix <<" has no emb_doc_map object found in the KV store" << std::endl;
            dbg_default_error("[{}]at {}, Failed to find object prefix {} in the KV store.", gettid(), __func__, table_prefix);
            return -1;
        }
        std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filtered_keys = filter_exact_matched_keys(map_obj_keys, table_prefix);
        // 1. get the doc table for the cluster_id
        while(!filtered_keys.empty()){
            std::string map_obj_key = filtered_keys.top();
            filtered_keys.pop();
            auto get_query_results = typed_ctxt->get_service_client_ref().get(map_obj_key);
            auto& reply = get_query_results.get().begin()->second.get();
            if (reply.blob.size == 0) {
                std::cerr << "Error: failed to get the doc table for key=" << map_obj_key << std::endl;
                dbg_default_error("Failed to get the doc table for key={}.", map_obj_key);
                return false;
            }
            char* json_data = const_cast<char*>(reinterpret_cast<const char*>(reply.blob.bytes));
            std::string json_str(json_data, reply.blob.size);
            try{
                nlohmann::json doc_table_json = nlohmann::json::parse(json_str);
                for (const auto& [emb_index, pathname] : doc_table_json.items()) {
                    this->doc_tables[cluster_id][std::stol(emb_index)] = "/rag/doc/" + std::to_string(pathname.get<int>());
                }
            } catch (const nlohmann::json::parse_error& e) {
                std::cerr << "Error: load_doc_table JSON parse error: " << e.what() << std::endl;
                dbg_default_error("{}, JSON parse error: {}", __func__, e.what());
                return false;
            }
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING     
        TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_END, my_id, 0, cluster_id);
#endif
        return true;
    }

    bool get_doc(DefaultCascadeContextType* typed_ctxt, int cluster_id, long emb_index, std::string& res_doc){
        if (doc_contents.find(cluster_id) != doc_contents.end()) {
            if (doc_contents[cluster_id].find(emb_index) != doc_contents[cluster_id].end()) {
                res_doc = doc_contents[cluster_id][emb_index];
                return true;
            }
        }
        bool loaded_doc_table = load_doc_table(typed_ctxt, cluster_id);
        if (!loaded_doc_table) {
            dbg_default_error("Failed to load the doc table for cluster_id={}.", cluster_id);
            return false;
        }
        if (doc_tables[cluster_id].find(emb_index) == doc_tables[cluster_id].end()) {
            std::cerr << "Error: failed to find the doc pathname for cluster_id=" << cluster_id << " and emb_id=" << emb_index << std::endl;
            dbg_default_error("Failed to find the doc pathname for cluster_id={} and emb_id={}, query={}.", cluster_id, emb_index);
            return false;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_START, this->my_id, emb_index, cluster_id);
#endif 
        auto& pathname = doc_tables[cluster_id][emb_index];

        if(SKIP_DOC_RETRIEVAL){
            res_doc = pathname;
            return true;
        }

        
        auto get_doc_results = typed_ctxt->get_service_client_ref().get(pathname);
        auto& reply = get_doc_results.get().begin()->second.get();
        if (reply.blob.size == 0) {
            std::cerr << "Error: failed to cascade get the doc content for pathname=" << pathname << std::endl;
            dbg_default_error("Failed to cascade get the doc content for pathname={}.", pathname);
            return false;
        }
        // parse the reply.blob.bytes to std::string
        char* doc_data = const_cast<char*>(reinterpret_cast<const char*>(reply.blob.bytes));
        std::string doc_str(doc_data, reply.blob.size);  /*** TODO: this is a copy, need to optimize */
        this->doc_contents[cluster_id][emb_index] = doc_str;
        res_doc = this->doc_contents[cluster_id][emb_index];
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_END, this->my_id, emb_index, cluster_id);
#endif
        return true;
    }


    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override { 
        // 0. parse the query information from the key_string
        int client_id, cluster_id, batch_id, qid;
        if (!parse_query_info(key_string, client_id, batch_id, cluster_id, qid)) {
            std::cerr << "Error: failed to parse the query_info from the key_string:" << key_string << std::endl;
            dbg_default_error("In {}, Failed to parse the query_info from the key_string:{}.", __func__, key_string);
            return;
        }
        
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        int query_batch_id = batch_id * 100000 + qid % 100000; // cast down qid for logging purpose
        TimestampLogger::log(LOG_TAG_AGG_UDL_START,client_id,query_batch_id,cluster_id);
#endif
        dbg_default_trace("[AggregateGenUDL] receive cluster search result from cluster{}.", cluster_id);
        std::string query_text;
        std::vector<DocIndex> cluster_results;
        // 1. deserialize the cluster searched result from the object
        try{
            deserialize_cluster_search_result_from_bytes(cluster_id, object.blob.bytes, object.blob.size, query_text, cluster_results);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to deserialize the cluster searched result and query texts from the object." << std::endl;
            dbg_default_error("{}, Failed to deserialize the cluster searched result from the object.", __func__);
            return;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_TAG_AGG_UDL_FINISHED_DESERIALIZE, client_id, query_batch_id, cluster_id);
#endif
        // 2. add the cluster_results to the query_results and check if all results are collected
        if (query_results.find(query_text) == query_results.end()) {
            query_results[query_text] = std::make_unique<ClusterSearchResults>(query_text, top_num_centroids, top_k);
        }
        bool added_cluster_result = query_results[query_text]->add_cluster_result(cluster_id, cluster_results);
        if (!added_cluster_result) {
            std::cerr << "Error: failed to add the cluster search result for query=" << query_text << " and cluster_id=" << cluster_id << std::endl;
            dbg_default_error("Failed to add the cluster search result for query={} and cluster_id={}.", query_text, cluster_id);
            return;
        }
        if (!query_results[query_text]->is_all_results_collected()) {
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_TAG_AGG_UDL_END_NOT_FULLY_GATHERED, client_id, query_batch_id, cluster_id);
#endif
            return;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_TAG_AGG_UDL_RETRIEVE_DOC_START, client_id, query_batch_id, cluster_id);
#endif
        // 3. All cluster results are collected for this query, aggregate the top_k results
        auto& agg_top_k_results = query_results[query_text]->agg_top_k_results;
        // 4. get the top_k docs content
        std::vector<std::string> top_k_docs(top_k);
        uint i = top_k - 1;
        while (!agg_top_k_results.empty()) {
            auto doc_index = agg_top_k_results.top();
            agg_top_k_results.pop();
            std::string res_doc;
            bool find_doc = get_doc(typed_ctxt,doc_index.cluster_id, doc_index.emb_id, res_doc);
            if (!find_doc) {
                std::cerr << "Error: failed to get_doc for cluster_id=" << doc_index.cluster_id << " and emb_id=" << doc_index.emb_id << std::endl;
                dbg_default_error("Failed to get_doc for cluster_id={} and emb_id={}.", doc_index.cluster_id, doc_index.emb_id);
                return;
            }
            top_k_docs[i--] = res_doc;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_TAG_AGG_UDL_RETRIEVE_DOC_END, client_id, query_batch_id, qid);
#endif
        // 5. run LLM with the query and its top_k closest docs

        // 6. put the result to cascade and notify the client
        // convert the query and top_k_docs to a json object
        nlohmann::json result_json;
        result_json["query"] = query_text;
        result_json["top_k_docs"] = top_k_docs;
        std::string result_json_str = result_json.dump();
        // put the result to cascade
        std::string result_key = "/rag/results/" + std::to_string(client_id) + "/" + std::to_string(qid);
        ObjectWithStringKey result_obj(result_key, reinterpret_cast<const uint8_t*>(result_json_str.c_str()), result_json_str.size());
        try {
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            /*** TODO: encode qid in result, so that the client could log the corresponding query ***/
            TimestampLogger::log(LOG_TAG_AGG_UDL_PUT_RESULT_START, client_id, query_batch_id, qid);
#endif
            std::string notification_pathname = "/rag/results/" + std::to_string(client_id);
            typed_ctxt->get_service_client_ref().notify(result_obj.blob,notification_pathname,client_id);
            dbg_default_trace("[AggregateGenUDL] echo back to node {}", client_id);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_TAG_AGG_UDL_PUT_RESULT_END, client_id, query_batch_id, qid);
#endif
            // 7. (garbage collection) remove query and query_result from the cache
            query_results.erase(query_text);
        } catch (derecho::derecho_exception& ex) {
            std::cerr << "[AGGnotification ocdpo]: exception on notification:" << ex.what() << std::endl;
            dbg_default_error("[AGGnotification ocdpo]: exception on notification:{}", ex.what());
        }
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<AggGenOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
        this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
        try{
            if (config.contains("top_num_centroids")) {
                this->top_num_centroids = config["top_num_centroids"].get<int>();
            }
            if (config.contains("final_top_k")) {
                this->top_k = config["final_top_k"].get<int>();
            }
            if (config.contains("include_llm")) {
                this->include_llm = config["include_llm"].get<bool>();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to convert top_num_centroids, top_k or include_llm from config" << std::endl;
            dbg_default_error("Failed to convert top_num_centroids, top_k or include_llm from config, at clusters_search_udl.");
        }
    }
};

std::shared_ptr<OffCriticalDataPathObserver> AggGenOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    AggGenOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext* ctxt,const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<AggGenOCDPO>(AggGenOCDPO::get())->set_config(typed_ctxt,config);
    return AggGenOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho
