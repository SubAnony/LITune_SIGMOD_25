#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <string>
#include <sstream>


using namespace std;


#define TIME_WINDOW 9000000
#define NO_BULK_LOAD 1000000
#define NO_INSERT 8000000
// #define NO_BULK_LOAD 500
// #define NO_INSERT 8000



#include "src/carmi_map.h"
#include "rdtsc.h"

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

template<typename K, typename V>
bool load_binary_sosd_key_value(string filename, vector<pair<K,V>> &v)
{
    vector<K> data;
    load_binary_sosd<K>(filename,data);
    if (!load_binary_sosd<K>(filename,data))
    {
        cout << "input stream status error" << endl;
        return false;
    }

    //Sort and Unique (if we want)
    sort(data.begin(),data.end());
    data.erase( unique( data.begin(), data.end() ), data.end());

    int cnt = 0;
    for(auto it = data.begin(); it != data.begin() + NO_BULK_LOAD + NO_INSERT; ++it)
    {
        v.push_back(make_pair(*it,static_cast<V>(cnt)));
        ++cnt;
    }
    return true;
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
            cout << input_query_type<<endl;
            cout << "Invalid query_type specified. Options: balanced, read-heavy, write-heavy (default: balanced)" << endl;
            return 1;
        }
    }

    // vector<uint64_t> data;
    // if (!load_binary_sosd(data_file_path, data))
    // {
    //     cout << "input stream status error" << endl;
    // }

    vector<pair<uint64_t,uint64_t>> dataKV;
    vector<pair<uint64_t,uint64_t>> update_data;
    vector<pair<uint64_t,uint64_t>> bulk_data;


    // Extract the first 9M elements


    if (!load_binary_sosd_key_value<uint64_t,uint64_t>(data_file_path, dataKV))
    {
        cout << "input stream status error" << endl;
    }


    // Uniformly sample 1/9 elements for bulk load
    size_t step = dataKV.size() / NO_BULK_LOAD;
    for (size_t i = 0; i < dataKV.size(); i += step) {
        bulk_data.push_back(dataKV[i]);
    }

    // Extract the remaining 8/9 elements for insertion
    for (const auto& kv : dataKV) {
        bool found = false;
        for (const auto& up_kv : update_data) {
            if (up_kv.first == kv.first) {
                found = true;
                break;
            }
        }
        if (!found) {
            update_data.push_back(make_pair(kv.first, 0));
        }
        }

    //Set up Lambda (check carmi_parameters.txt)
    float lambda;
    if (external_lambda == -1)
    {
        float datasize = sizeof(uint64_t) *
                        (dataKV.size() + dataKV.size()) / 1024.0 /
                        1024.0;  // MB
        float leafRate = 1 + 64.0 / (carmi_params::kMaxLeafNodeSize * 0.75);
        lambda = 100.0 / leafRate / datasize;
    }
    else
    {
        lambda = external_lambda;
    }

    //Load CARMI (Takes a very long time, be patient)
    CARMIMap<uint64_t,uint64_t> carmi(bulk_data.begin(),bulk_data.end(),bulk_data.begin(),bulk_data.end(),lambda);
    

    clock_t start,end;

    auto it = update_data.begin();
    auto itDelete = update_data.begin();

    start=clock();


    //Run Query



    if (query_type == "balanced") {
    // Insert the rest 4M elements
    for (int i = 0; i < 4000000; ++i)
    {
        carmi.insert(*it);
        ++it;
    }
    // Range query 8M
    for (int i = 0; i < 4000000; ++i) {
        uint64_t query_key = dataKV.at((dataKV.begin() - dataKV.begin()) + (rand() % 
        ( (dataKV.end() - dataKV.begin()) - (dataKV.begin() - dataKV.begin()) + 1 ))).first;

        carmi.lower_bound(query_key);
        carmi.upper_bound(query_key);
        // carmi.find(query_key);
        // carmi.erase(query_key);
    }

    //Delete query 4M

    for (int i = 0; i < 4000000; ++i)
    {

        carmi.erase(itDelete->first);
        ++itDelete;
    }   
    } else if (query_type == "read-heavy") {
    //Read-heavy query

    // Insert 4M elements
    for (int i = 0; i < 4000000; ++i)
    {
        carmi.insert(*it);
        ++it;
    }

    // Range query, 24M

    for (int i = 0; i < 12000000; ++i) {
        uint64_t query_key = dataKV.at((dataKV.begin() - dataKV.begin()) + (rand() % 
        ( (dataKV.end() - dataKV.begin()) - (dataKV.begin() - dataKV.begin()) + 1 ))).first;

        carmi.lower_bound(query_key);
        carmi.upper_bound(query_key);
        // carmi.find(query_key);
        // carmi.count(query_key);
        // carmi.erase(query_key);
    }
    
    // Delete 4M elements
    for (int i = 0; i < 4000000; ++i)
    {

        carmi.erase(itDelete->first);
        ++itDelete;
    }

    } else if (query_type == "write-heavy") {
    //Write-heavy query

    // Insert 6M elements
    for (int i = 0; i < 6000000; ++i)
    {
        carmi.insert(*it);
        ++it;
    }

    // Range query, 4M

    for (int i = 0; i < 1000000; ++i) {
        uint64_t query_key = dataKV.at((dataKV.begin() - dataKV.begin()) + (rand() % 
        ( (dataKV.end() - dataKV.begin()) - (dataKV.begin() - dataKV.begin()) + 1 ))).first;

        carmi.lower_bound(query_key);
        carmi.upper_bound(query_key);
        carmi.find(query_key);
        carmi.count(query_key);
        // carmi.erase(query_key);
    }
    
    // Delete 6M elements
    for (int i = 0; i < 6000000; ++i)
    {

        carmi.erase(itDelete->first);
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

    //Output execution time
    ofstream run_time_out("./runtime_result.txt");
    assert(run_time_out);
    run_time_out << (double)(end-start)/CLOCKS_PER_SEC << endl;
    run_time_out.close();
    if (!run_time_out.good())
    {
        cout << "runtime_result out stream status error" << endl;
    }

    //Ouput States
    ofstream  state_out("./state_result.txt");
    assert(state_out);

    state_out << "no_leaf:" << carmi.carmi_tree.lastLeaf - carmi.carmi_tree.firstLeaf + 1<< endl;
    
    state_out << "lambda:" << carmi.carmi_tree.lambda << endl;

    state_out << "prefetchEnd:" << carmi.carmi_tree.prefetchEnd << endl;

    state_out << "querySize:" << carmi.carmi_tree.querySize << endl;

    state_out << "reservedSpace:" << carmi.carmi_tree.reservedSpace << endl;

    state_out << "isInitMode:" << carmi.carmi_tree.isInitMode << endl;

    vector<pair<uint64_t,int>> findQuery =  carmi.carmi_tree.stats_findQuery;
    state_out << "noFindQueryKeyVisit:" <<  findQuery.size() << endl;
    double avgfindQueryVisitPerKey = 0;
    for (auto &it: findQuery)
    {
        avgfindQueryVisitPerKey += it.second;
    }
    avgfindQueryVisitPerKey /= findQuery.size();
    state_out << "avgfindQueryVisitPerKey:" <<  avgfindQueryVisitPerKey << endl;

    vector<uint64_t> insertQuery = carmi.carmi_tree.stats_insertQuery;
    state_out << "noInsertQueryKeyVisit:" << insertQuery.size() << endl;

    map<IndexPair, NodeCost> cost = carmi.carmi_tree.stats_COST;
    state_out << "costSize:" << cost.size() << endl;
    double avgTime = 0;
    double avgSpace = 0;
    double avgTotal = 0;
    for (auto &it: cost)
    {
        avgTime += it.second.time;
        avgSpace += it.second.space;
        avgTotal += it.second.cost;
    }
    avgTime /= cost.size();
    avgSpace /= cost.size();
    avgTotal /= cost.size();
    state_out << "avgTimeCost:" << avgTime << endl;
    state_out << "avgSpaceCost:" << avgSpace << endl;
    state_out << "avgTotalCost:" << avgTotal << endl;
    
    vector<int> remainingNode = carmi.carmi_tree.stats_remainingNode;
    state_out << "remainingNode:" << remainingNode.size() << endl;

    vector<DataRange> remainingRange = carmi.carmi_tree.stats_remainingRange;
    state_out << "scanLeafRange:" << remainingRange.size() << endl;
    state_out.close();
    if (!state_out.good())
    {
        cout << "state_result out stream status error" << endl;
    }

    return 0;
}