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

// #define DATA_DIR "/data/Documents/data/"
// #define FILE_NAME "f_osm"
#define TIME_WINDOW 100000
#define NO_INSERT 1000
#define NO_DELETE 1000
#define NO_SEARCH 4000
#define MACHINE_FREQUENCY 3400000000

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
    for(auto it = data.begin(); it != data.begin() + TIME_WINDOW + NO_INSERT; ++it)
    {
        v.push_back(make_pair(*it,static_cast<V>(cnt)));
        ++cnt;
    }
    return true;
}

int main(int argc, char * argv[]) 
{
    //SOSD data path
    // string pathFileName = DATA_DIR FILE_NAME;
    string pathFileName = argv[1];

    //Load Data
    vector<pair<uint64_t,uint64_t>> dataKV;
    if (!load_binary_sosd_key_value<uint64_t,uint64_t>(pathFileName,dataKV))
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
    


    //Setup for Running Query
    auto it = dataKV.begin()+TIME_WINDOW;
    auto itDelete = dataKV.begin();

    int resultCount = 0;

    uint64_t searchCycle = 0;
    uint64_t insertCycle = 0;
    uint64_t deleteCycle = 0;

    //Run Query
    for (int i = 0; i < NO_INSERT; ++i)
    {
        uint64_t tempInsertCycles = 0;
        startTimer(&tempInsertCycles);
        carmi.insert(*it);
        stopTimer(&tempInsertCycles);
        insertCycle += tempInsertCycles;

        ++it;
    }

    for (int i = 0; i < NO_SEARCH; ++i)
    {
        uint64_t query_key = dataKV.at((itDelete - dataKV.begin()) + (rand() % ( (it - dataKV.begin()) - (itDelete - dataKV.begin()) + 1 ))).first;
        
        uint64_t tempSearchCycles = 0;
        startTimer(&tempSearchCycles);
        auto it = carmi.find(query_key);
        stopTimer(&tempSearchCycles);
        searchCycle += tempSearchCycles;

        resultCount = (it != carmi.end())? resultCount + 1: resultCount;
    }

    for (int i = 0; i < NO_DELETE; ++i)
    {
        uint64_t tempDeleteCycles = 0;
        startTimer(&tempDeleteCycles);
        carmi.erase(itDelete->first);
        stopTimer(&tempDeleteCycles);
        deleteCycle += tempDeleteCycles;
        ++itDelete;
    }

    uint64_t totalCycle = searchCycle + insertCycle + deleteCycle;

    //Output execution time
    ofstream run_time_out("./runtime_result.txt");
    assert(run_time_out);

    //If you want details:
    run_time_out << "searchTime:" << static_cast<double>(searchCycle)/MACHINE_FREQUENCY << endl;
    run_time_out << "insertTime:" << static_cast<double>(insertCycle)/MACHINE_FREQUENCY << endl;
    run_time_out << "deleteTime:" << static_cast<double>(deleteCycle)/MACHINE_FREQUENCY << endl;

    //If you want single number (comment above)
    run_time_out << "totalTime:" << static_cast<double>(totalCycle)/MACHINE_FREQUENCY << endl;

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
    state_out << "noInsertQueryKeyVisit:" << carmi.carmi_tree.stats_insertQuery.size() << endl;
    state_out << "remainingNode:" << carmi.carmi_tree.stats_remainingNode.size() << endl;
    state_out << "scanLeafRange:" << carmi.carmi_tree.stats_remainingRange.size() << endl;

    state_out << "noFindQueryKeyVisit:" <<  carmi.carmi_tree.stats_findQuery.size() << endl;
    double avgfindQueryVisitPerKey = 0;
    for (auto &it: carmi.carmi_tree.stats_findQuery)
    {
        avgfindQueryVisitPerKey += it.second;
    }
    avgfindQueryVisitPerKey /= carmi.carmi_tree.stats_findQuery.size();
    state_out << "avgfindQueryVisitPerKey:" <<  avgfindQueryVisitPerKey << endl;

    state_out << "costSize:" << carmi.carmi_tree.stats_COST.size() << endl;
    double avgTime = 0;
    double avgSpace = 0;
    double avgTotal = 0;
    for (auto &it: carmi.carmi_tree.stats_COST)
    {
        avgTime += it.second.time;
        avgSpace += it.second.space;
        avgTotal += it.second.cost;
    }
    avgTime /= carmi.carmi_tree.stats_COST.size();
    avgSpace /= carmi.carmi_tree.stats_COST.size();
    avgTotal /= carmi.carmi_tree.stats_COST.size();
    state_out << "avgTimeCost:" << avgTime << endl;
    state_out << "avgSpaceCost:" << avgSpace << endl;
    state_out << "avgTotalCost:" << avgTotal << endl;


    state_out.close();
    if (!state_out.good())
    {
        cout << "state_result out stream status error" << endl;
    }
    return 0;
}