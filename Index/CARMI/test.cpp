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

#include "src/carmi_map.h"

#define DATA_DIR "/data/Documents/data/"
#define FILE_NAME "f_osm"
#define TIME_WINDOW 1000 //9000000

template<typename K>
bool load_binary_sosd(string filename, vector<K> &v)
{
    ifstream ifs(filename, ios::in | ios::binary);
    assert(ifs);

    K size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(K));
    v.resize(size);
    ifs.read(reinterpret_cast<char*>(v.data()), size * sizeof(K));
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
    for(auto it = data.begin(); it != data.begin() + TIME_WINDOW; ++it)
    {
        v.push_back(make_pair(*it,static_cast<V>(cnt)));
        ++cnt;
    }
    return true;
}

int main(int argc, char * argv[]) 
{
    //SOSD data path
    // string filename = argv[1];
    string filename = argv[1];
    
    //Load Data
    vector<pair<uint64_t,uint64_t>> dataKV;
    if (!load_binary_sosd_key_value<uint64_t,uint64_t>(filename,dataKV))
    {
        cout << "input stream status error" << endl;
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
    CARMIMap<uint64_t,uint64_t> carmi(dataKV.begin(),dataKV.end(),dataKV.begin(),dataKV.end(),lambda);

    // Run Query 
    clock_t start,end;
    start=clock();
    for (int i = 0; i < 100; ++i)
    {
        uint64_t query_key = dataKV.at((dataKV.begin() - dataKV.begin()) + (rand() % 
        ( (dataKV.end() - dataKV.begin()) - (dataKV.begin() - dataKV.begin()) + 1 ))).first;

        carmi.count(query_key);
        carmi.find(query_key);
        carmi.lower_bound(query_key);
        carmi.upper_bound(query_key);
    }
    end=clock();
    

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