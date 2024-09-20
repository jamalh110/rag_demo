#include <cascade/service_client_api.hpp>
#include <chrono>
#include <filesystem> 
#include <iostream>
#include <unistd.h>  
#include <vector>

using namespace derecho::cascade;
#define EMBEDDING_DIM 960
#define VORTEX_SUBGROUP_INDEX 0
#define AGG_SUBGROUP_INDEX 0
#define QUERY_FILENAME "query.csv"
#define QUERY_EMB_FILENAME "query_emb.csv"

// Use vector since one query may be reuse for multiple times
std::unordered_map<std::string, std::vector<std::tuple<int, int>>> sent_queries;
std::unordered_map<std::string, std::string> query_results;

int read_queries(std::filesystem::path query_filepath, int num_queries, int batch_size, std::vector<std::string>& queries) {
     std::ifstream file(query_filepath);
     if (!file.is_open()) {
          std::cerr << "Error: Could not open query directory:" << query_filepath << std::endl;
          std::cerr << "Current only support query_doc in csv format." << std::endl;
          return 0;
     }
     std::string line;
     int num_query_collected = 0;
     while (std::getline(file, line)) {
          if (num_query_collected >= num_queries * batch_size) {
               break;
          }
          queries.push_back(line);
          num_query_collected++;
     }
     file.close();
     return num_query_collected;
}

int read_query_embs(std::string query_emb_directory, int num_queries, int batch_size, std::unique_ptr<float[]>& query_embs){
     std::unique_ptr<float[]> embs = std::make_unique<float[]>(EMBEDDING_DIM * num_queries * batch_size);
     std::ifstream file(query_emb_directory);
     if (!file.is_open()) {
          std::cerr << "Error: Could not open query directory:" << query_emb_directory << std::endl;
          std::cerr << "Current only support query_doc in csv format." << std::endl;
          return 0;
     }
     std::string line;
     int num_query_collected = 0;
     while (std::getline(file, line)) {
          if (num_query_collected >= num_queries * batch_size) {
               break;
          }
          std::istringstream ss(line);
          std::string token;
          int i = 0;
          while (std::getline(ss, token, ',')) {
               embs[num_query_collected * EMBEDDING_DIM + i] = std::stof(token);
               i++;
          }
          if (i != EMBEDDING_DIM) {
               std::cerr << "Error: query embedding dimension does not match." << std::endl;
               return 0;
          }
          num_query_collected++;
     }
     file.close();
     // resize the embs array to the actual number of queries collected
     if (num_query_collected < num_queries * batch_size) {
          std::unique_ptr<float[]> new_embs = std::make_unique<float[]>(EMBEDDING_DIM * num_query_collected);
          std::memcpy(new_embs.get(), embs.get(), EMBEDDING_DIM * num_query_collected * sizeof(float));
          embs = std::move(new_embs);
     }
     query_embs = std::move(embs);
     return num_query_collected;
}


std::string format_query_emb_object(int nq, std::unique_ptr<float[]>& xq, std::vector<std::string>& query_list) {
     // create an bytes object by concatenating: num_queries + float array of emebddings + list of query_text
     uint32_t num_queries = static_cast<uint32_t>(nq);
     std::string nq_bytes(4, '\0');
     nq_bytes[0] = (num_queries >> 24) & 0xFF;
     nq_bytes[1] = (num_queries >> 16) & 0xFF;
     nq_bytes[2] = (num_queries >> 8) & 0xFF;
     nq_bytes[3] = num_queries & 0xFF;
     float* query_embeddings = xq.get();
     // serialize the query embeddings and query texts, formated as num_queries + query_embeddings + query_texts
     std::string query_emb_string = nq_bytes +
                              std::string(reinterpret_cast<const char*>(query_embeddings), sizeof(float) * EMBEDDING_DIM * num_queries) +
                              nlohmann::json(query_list).dump();
     return query_emb_string;
}

/***
 * Result JSON is in format of : {"query": query_text, "top_k_docs":[doc_text1, doc_text2, ...]}
 */
bool deserialize_result(const Blob& blob, std::string& query_text, std::vector<std::string>& top_k_docs) {
     if (blob.size == 0) {
          std::cerr << "Error: empty result blob." << std::endl;
          return false;
     }
     char* json_data = const_cast<char*>(reinterpret_cast<const char*>(blob.bytes));
     std::size_t json_size = blob.size;
     std::string json_string(json_data, json_size);
     try{
          nlohmann::json parsed_json = nlohmann::json::parse(json_string);
          if (parsed_json.count("query") == 0 || parsed_json.count("top_k_docs") == 0) {
               std::cerr << "Result JSON does not contain query or top_k_docs." << std::endl;
               return false;
          }
          query_text = parsed_json["query"];
          top_k_docs = parsed_json["top_k_docs"];

                  std::cout << "Top K Docs:" << std::endl;
        for (const auto& doc : top_k_docs) {
            std::cout << doc << std::endl;
}

     } catch (const nlohmann::json::parse_error& e) {
          std::cerr << "Result JSON parse error: " << e.what() << std::endl;
          return false;
     }
     return true;
}


bool run_latency_test(ServiceClientAPI& capi, int num_queries, int batch_size, std::string& query_directory, int query_interval) {
     int my_id = capi.get_my_id();
     // 1.1. Prepare for the notification by creating object pool for results to store
     std::string result_pool_name = "/rag/results/" + std::to_string(my_id);
     auto res = capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(result_pool_name,0,HASH,{});
     for (auto& reply_future:res.get()) {
          reply_future.second.get(); // wait for the object pool to be created
     }
     std::cout << "Created object pool for results: " << result_pool_name << std::endl;
     // 1.2. Register notification for this object pool
     std::atomic<bool> running(true);
     std::atomic<int> num_queries_to_send(num_queries);
     bool ret = capi.register_notification_handler(
               [&](const Blob& result){
                    std::string query_text;
                    std::vector<std::string> top_k_docs;
                    if (!deserialize_result(result, query_text, top_k_docs)) {
                         std::cerr << "Error: failed to deserialize the result from the notification." << std::endl;
                         return false;
                    }
                    if (query_results.find(query_text) == query_results.end()) {
                         query_results[query_text] = top_k_docs[0];
                    }
                    if (sent_queries.find(query_text) != sent_queries.end()) {
                         for (auto& [qb_id, q_id]: sent_queries[query_text]) {
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                              TimestampLogger::log(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED,my_id,qb_id,q_id);
#endif
                              // std::cout << "Received result for query: " << query_text << " from client: " << my_id << " qb_id: " << qb_id << " q_id: " << q_id << std::endl;
                              // remove [qb_id, q_id] from sent_queries
                              sent_queries[query_text].erase(std::remove(sent_queries[query_text].begin(), sent_queries[query_text].end(), std::make_tuple(qb_id, q_id)), sent_queries[query_text].end());
                              if (sent_queries[query_text].size() == 0) {
                                   sent_queries.erase(query_text);
                              }
                              break;
                         }
                    } else {
                         std::cerr << "Error: received result for query that is not sent." << std::endl;
                    }
                    if (sent_queries.size() == 0 && num_queries_to_send.load() == 0) {
                         running = false;
                         // std::cout << "Received all results." << std::endl;
                    }
                    return true;
               }, result_pool_name);
     if (ret) {
          std::cerr << "Replaced previous notification handler for obj_pool: " << result_pool_name << std::endl;
     }

     // 1.3. Establishing connections to all server nodes
     auto subgroup_members = capi.template get_subgroup_members<VolatileCascadeStoreWithStringKey>(AGG_SUBGROUP_INDEX);
     int num_shards = subgroup_members.size();
     std::string establish_key = "establish_connection";
     for (int i = 0; i < num_shards; i++) {
          ObjectWithStringKey obj;
          std::string control_value = "establish";
          obj.key = establish_key;
          obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
          auto res = capi.template put<VolatileCascadeStoreWithStringKey>(obj, AGG_SUBGROUP_INDEX, i, true);
          for (auto& reply_future:res.get()) {
               reply_future.second.get(); // wait for the object has been put
          }
     }
     std::cout << "Registered server-side notifications."<< std::endl;


     // 2. Prepare the query and query embeddings
     /** TODO: quite some copies in this process, not on critical path, but could be optimized. */
     std::filesystem::path query_pathname = std::filesystem::path(query_directory) / QUERY_FILENAME;
     std::vector<std::string> queries;
     int num_query_collected = read_queries(query_pathname, num_queries, batch_size, queries);
     if (queries.size() == 0) {
          std::cerr << "Error: failed to read queries from " << query_directory << std::endl;
          return false;
     }

     std::filesystem::path query_emb_pathname = std::filesystem::path(query_directory) / QUERY_EMB_FILENAME;
     std::unique_ptr<float[]> query_embs;
     int num_emb_collected = read_query_embs(query_emb_pathname, num_queries, batch_size, query_embs);
     // resize query or query_embs to make sure they correspond, i.e. have the same number of queries
     if (num_query_collected > num_emb_collected){ // truncate query_vector to match the number of query embeddings
          queries.resize(num_query_collected);
          num_query_collected = num_emb_collected;
     } else if (num_emb_collected > num_query_collected) { // truncate query_embs to match the number of queries
          std::unique_ptr<float[]> new_embs = std::make_unique<float[]>(EMBEDDING_DIM * num_query_collected);
          std::memcpy(new_embs.get(), query_embs.get(), EMBEDDING_DIM * num_emb_collected * sizeof(float));
          query_embs = std::move(new_embs);
          num_emb_collected = num_query_collected;
     }
     

     // 3. send the queries to the cascade
     for (int qb_id = 0; qb_id < num_queries; qb_id++) {
          std::string key = "/rag/emb/centroids_search/client" + std::to_string(my_id) + "/qb" + std::to_string(qb_id);
          /*** TODO: current method of prepare query text and emb cause extra copies that could be avoided. ***/
          // 3.1. Prepare the query texts
          std::vector<std::string> cur_query_list;
          for (int j = 0; j < batch_size; ++j) {
              cur_query_list.push_back(queries[(qb_id + j) % num_query_collected]);
              if (sent_queries.find(queries[(qb_id + j) % num_query_collected]) == sent_queries.end()) {
                   sent_queries[queries[(qb_id + j) % num_query_collected]] = std::vector<std::tuple<int, int>>();
              }
              sent_queries[queries[(qb_id + j) % num_query_collected]].push_back(std::make_tuple(qb_id, j));
          }
          // 3.2. Prepare query embeddings
          std::unique_ptr<float[]> query_embeddings(new float[EMBEDDING_DIM * batch_size]);
          for (int j = 0; j < batch_size; ++j) {
               int pos = (qb_id * batch_size + j) % num_emb_collected;
               for (int k = 0; k < EMBEDDING_DIM; ++k) {
                    query_embeddings[j * EMBEDDING_DIM + k] = query_embs[pos * EMBEDDING_DIM + k];
               }
          }
          // 3.3. format the query 
          std::string emb_query_string = format_query_emb_object(batch_size, query_embeddings, cur_query_list);
          ObjectWithStringKey emb_query_obj;
          emb_query_obj.key = key;
          emb_query_obj.blob = Blob(reinterpret_cast<const uint8_t*>(emb_query_string.c_str()), emb_query_string.size());
          // 3.4. send the object to the cascade
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
          for (int j = 0; j < batch_size; ++j) {
               TimestampLogger::log(LOG_TAG_QUERIES_SENDING_START,my_id,qb_id,j);
          }
#endif
          capi.put_and_forget(emb_query_obj, false); // not only trigger the UDL, but also update state. TODO: Need more thinking here. 
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
          for (int j = 0; j < batch_size; ++j) {
               TimestampLogger::log(LOG_TAG_QUERIES_SENDING_END,my_id,qb_id,j);
          }
#endif
          num_queries_to_send -= batch_size;
          std::this_thread::sleep_for(std::chrono::microseconds(query_interval));
          // // implement sleep using busy waiting and while loop
          // auto start = std::chrono::high_resolution_clock::now();
          // while (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() < query_interval) {
          //      // busy waiting
          // }
          if (qb_id % 200 == 0) {
               std::cout << "Sent " << qb_id << " queries." << std::endl;
          }
     }
     
     std::cout << "Put all queries to cascade." << std::endl;
     // std::this_thread::sleep_for(std::chrono::seconds(10));
     // running = false;
     // message_thread.join();
     while (running.load()) {
          std::this_thread::sleep_for(std::chrono::microseconds(query_interval/100));
     }

     // 4. flush logs
     std::string flush_log_key = "/rag/emb/centroids_search/flush_logs";
     for (int i = 0; i < num_shards; i++){
          ObjectWithStringKey obj;
          std::string control_value = "flush";
          obj.key = flush_log_key;
          obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
          auto res = capi.template put<VolatileCascadeStoreWithStringKey>(obj, VORTEX_SUBGROUP_INDEX, i, true);
          for (auto& reply_future:res.get()) {
               reply_future.second.get(); // wait for the object pool to be created
          }
     }
     std::cout << "Flushed logs to shards." << std::endl;
     TimestampLogger::flush("client_timestamp.dat");
     return true;
}


/*** TODO: handle the segmentation fault if call multiple times this, while servers not stoped and refreshed ***/
int main(int argc, char** argv){
     int opt;
     int num_queries = 0;
     int batch_size = 0;
     // int query_interval = 100000; // default interval between query is 1 second
     int query_interval = 50000;
     std::string query_directory = "";

     while ((opt = getopt(argc, argv, "n:b:q:i:")) != -1) {
          switch (opt) {
               case 'n':
                    num_queries = std::atoi(optarg);  // Convert the argument to an integer
                    break;
               case 'b':
                    batch_size = std::atoi(optarg);   // Convert the argument to an integer
                    break;
               case 'q':
                    query_directory = optarg;
                    break;
               case 'i':
                    query_interval = std::atoi(optarg);
                    break;
               case '?': // Unknown option or missing option argument
                    std::cerr << "Usage: " << argv[0] << " -n <number_of_queries> -b <batch_size> -q <query_data_dir> -i <interval>" << std::endl;
                    return 1;
               default:
                    break;
          }
     }
     if (num_queries == 0 || batch_size == 0 || query_directory.empty()) {
          std::cerr << "Error: Missing required options." << std::endl;
          std::cerr << "Usage: " << argv[0] << " -n <number_of_queries> -b <batch_size> -q <query_dir.csv> -i <interval>" << std::endl;
          return 1;
     }

     std::cout << "Number of queries: " << num_queries << std::endl;
     std::cout << "Batch size: " << batch_size << std::endl;

     auto& capi = ServiceClientAPI::get_service_client();
     run_latency_test(capi, num_queries, batch_size, query_directory, query_interval);

     return 0;
}
