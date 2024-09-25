#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <cassert>
#include <string>
#include <sstream>
#include <random>

using namespace std;

#include "src/alex_map.h"

#define TIME_WINDOW 9000000
#define NO_BULK_LOAD 1000000
#define NO_INSERT 8000000


#define NO_SEARCH 6000000
#define TOTAL_PAIRS 8000000*3
// #define TIME_WINDOW 9000000
// #define TIME_WINDOW 500000



template<typename T>
bool load_binary_sosd(string filename, vector<T> &v)
{
    ifstream ifs(filename, ios::in | ios::binary);
    assert(ifs);

    T size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(T));
    v.resize(size);
    ifs.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
    ifs.close();

    return ifs.good();
}

int main(int argc, char * argv[])
{

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <data_file_path> [query_type]" << endl;
        cout << "query_type options: balanced, read-heavy, write-heavy (default: balanced)" << endl;
        return 1;
    }

    string data_file_path = argv[1];
    string query_type = "balanced";  // set default query_type to balanced

    if (argc > 2) {
        string input_query_type = argv[2];
        if (input_query_type == "balanced" || input_query_type == "read-heavy" || input_query_type == "write-heavy") {
            query_type = input_query_type;
        } else {
            cout << "Invalid query_type specified. Options: balanced, read-heavy, write-heavy (default: balanced)" << endl;
            return 1;
        }
    }


    vector<uint64_t> data;
    if (!load_binary_sosd(data_file_path, data))
    {
        cout << "input stream status error" << endl;
    }

    // Sort the data
    sort(data.begin(), data.end());

    // Extract the first 10M elements
    vector<uint64_t> first_9M_data(data.begin(), data.begin() + NO_BULK_LOAD + NO_INSERT);

    // Uniformly sample 1M elements for bulk load
    vector<uint64_t> bulk_load_data;
    vector<uint64_t> update_data;

    size_t step = first_9M_data.size() / NO_BULK_LOAD;
    for (size_t i = 0; i < first_9M_data.size(); i += step) {
        bulk_load_data.push_back(first_9M_data[i]);
    }

    // Extract the remaining 9M elements for insertion
    size_t update_idx = 0;
    for (size_t i = 0; i < first_9M_data.size(); ++i) {
        if (i == bulk_load_data[update_idx]) {
            update_idx++;
        } else {
            update_data.push_back(first_9M_data[i]);
        }
    }

    // Load ALEX
    alex::Alex<uint64_t, int> alex;

    // Bulk load
    // vector<pair<uint64_t, int>> dataArray;
    // for (size_t i = 0; i < bulk_load_data.size(); ++i)
    // {
    //     dataArray.push_back(make_pair(bulk_load_data[i], i));
    // }
    // alex.bulk_load(dataArray.data(), dataArray.size());

    // Load into ALEX (no bulk load), underlying distribution
    for (int i = 0; i < bulk_load_data.size(); ++i)
    {
        alex.insert(bulk_load_data[i],i);
    }


    clock_t start,end;

    auto it = update_data.begin();
    auto itDelete = update_data.begin();

    start=clock();

    if (query_type == "balanced") {
    // Insert the rest 8M elements
    for (int i = 0; i < update_data.size(); ++i)
    {
        alex.insert(*it, i);
        ++it;
    }

    // Range query 16M

    for (int i = 0; i < update_data.size(); ++i) {
        uint64_t query_key = update_data[i];
        alex.lower_bound(query_key);
        alex.upper_bound(query_key);
        // alex.find(query_key);
        // alex.erase(query_key);
    }

    //Delete query 8M

    for (int i = 0; i < update_data.size(); ++i)
    {

        alex.erase(*itDelete);
        ++itDelete;
    }   
    } 
    
    
    
    
    else if (query_type == "read-heavy") {
    //Read-heavy query

    // Insert 4M elements
    for (int i = 0; i < 4000000; ++i)
    {
        alex.insert(*it, i);
        ++it;
    }

    // Range query, 24M

    for (int i = 0; i < NO_SEARCH; ++i) {
        uint64_t query_key = update_data[i];
        alex.lower_bound(query_key);
        alex.upper_bound(query_key);
        alex.find(query_key);
        alex.count(query_key);
        // alex.erase(query_key);
    }
    
    // Delete 4M elements
    for (int i = 0; i < 4000000; ++i)
    {

        alex.erase(*itDelete);
        ++itDelete;
    }

    } 



    else if (query_type == "write-heavy") {
    //Write-heavy query

    // Insert 6M elements
    for (int i = 0; i < 6000000; ++i)
    {
        alex.insert(*it, i);
        ++it;
    }

    // Range query, 4M

    for (int i = 0; i < 1000000; ++i) {
        uint64_t query_key = update_data[i];
        alex.lower_bound(query_key);
        alex.upper_bound(query_key);
        alex.find(query_key);
        alex.count(query_key);
        // alex.erase(query_key);
    }
    
    // Delete 6M elements
    for (int i = 0; i < 6000000; ++i)
    {

        alex.erase(*itDelete);
        ++itDelete;
    }
    }
    end=clock();
    double total_time = (double)(end-start)/CLOCKS_PER_SEC;

    double throughput = 0.0;

    if (query_type == "balanced") {
        throughput = (4.0 * update_data.size()) / total_time;
    }
    else if (query_type == "read-heavy") {
        throughput = (32000000.0) / total_time; 
        }
    else {
        throughput = (16000000.0) / total_time;
    }

    //Output into File
    ofstream throughput_out("./throughput_result.txt");
    assert(throughput_out);
    // cout<< "Throughput: " << throughput << " ops/sec." << endl;
    throughput_out << "Throughput: " << throughput << " ops/sec." << endl;
    throughput_out.close();
  

    //Output into File
    ofstream run_time_out("./runtime_result.txt");
    assert(run_time_out);
    run_time_out << (double)(end-start)/CLOCKS_PER_SEC << endl;
    run_time_out.close();
    if (!run_time_out.good())
    {
        cout << "runtime_result out stream status error" << endl;
    }

    ofstream  state_out("./state_result.txt");
    assert(state_out);
    state_out << "no_model_nodes:" << alex.stats_.num_model_nodes << endl;
    state_out << "no_model_node_expansions:" << alex.stats_.num_model_node_expansions << endl;
    state_out << "no_model_node_split:" << alex.stats_.num_model_node_splits << endl;
    state_out << "num_model_node_expansion_pointers:" << alex.stats_.num_model_node_expansion_pointers << endl;
    state_out << "num_model_node_split_pointers:" << alex.stats_.num_model_node_split_pointers << endl;
    state_out << "no_data_nodes:" << alex.stats_.num_data_nodes << endl;
    state_out << "no_expand_and_scale:" << alex.stats_.num_expand_and_scales << endl;
    state_out << "no_expand_and_retrain:" << alex.stats_.num_expand_and_retrains << endl;
    state_out << "no_downward_split:" << alex.stats_.num_downward_splits << endl;
    state_out << "no_sideways_split:" << alex.stats_.num_sideways_splits << endl;
    state_out << "no_downward_split_keys:" << alex.stats_.num_downward_split_keys << endl;
    state_out << "no_sideways_split_keys:" << alex.stats_.num_sideways_split_keys << endl;
    state_out << "no_search:" << alex.stats_.num_lookups << endl;
    state_out << "no_inserts:" << alex.stats_.num_inserts << endl;
    state_out << "no_node_traveral:" << alex.stats_.num_node_lookups << endl;
    state_out.close();
    if (!state_out.good())
    {
        cout << "state_result out stream status error" << endl;
    }


    return 0;
}
