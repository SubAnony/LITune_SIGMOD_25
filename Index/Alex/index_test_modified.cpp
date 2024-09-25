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

#define TIME_WINDOW 900000
#define NO_BULK_LOAD 1000000
#define NO_INSERT 8000000
#define NO_DELETE 250000
#define NO_SEARCH 400000
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
    string data_file_path = argv[1];

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
    vector<pair<uint64_t, int>> dataArray;
    for (size_t i = 0; i < bulk_load_data.size(); ++i)
    {
        dataArray.push_back(make_pair(bulk_load_data[i], i));
    }
    alex.bulk_load(dataArray.data(), dataArray.size());

    // Load into ALEX (no bulk load), underlying distribution
    // for (int i = 0; i < bulk_load_data.size(); ++i)
    // {
    //     alex.insert(bulk_load_data[i],i);
    // }


    clock_t start,end;

    auto it = update_data.begin();
    auto itDelete = update_data.begin();

    start=clock();
    // Insert the rest 9M elements
    for (int i = 0; i < update_data.size(); ++i)
    {
        alex.insert(*it, i);
        ++it;
    }

    // Range query and delete
    for (int i = 0; i < update_data.size(); ++i) {
        uint64_t query_key = update_data[i];
        alex.lower_bound(query_key);
        alex.upper_bound(query_key);
        // alex.erase(query_key);
    }

    for (int i = 0; i < update_data.size(); ++i)
    {

        alex.erase(*itDelete);
        ++itDelete;
    }

    end=clock();

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
