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

#include "src/alex_map.h"
// #include "src/alex_multimap.h"

#define DATA_DIR "../data/"
#define FILE_NAME "books_200M_uint64"
#define TIME_WINDOW 9000000
#define NO_INSERT 100000
#define NO_DELETE 100000
#define NO_SEARCH 400000
#define MACHINE_FREQUENCY 3400000000

#include "src/alex_map.h"
// #include "src/alex_multimap.h"
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

int main(int argc, char * argv[]) 
{
    //SOSD data path
    string data_file_path = argv[1];
    
    //Load Data
    vector<uint64_t> data;
    if (!load_binary_sosd(data_file_path,data))
    {
        cout << "input stream status error" << endl;
    }

    //If we want to sort the data
    sort(data.begin(),data.end());
    data.erase( unique( data.begin(), data.end() ), data.end() );

    //Load ALEX
    alex::Alex<uint64_t,int> alex;

    //Load into ALEX (No Bulk Load)
    for (int i = 0; i < TIME_WINDOW; ++i)
    {
        alex.insert(data[i],i);
    }

    auto it = data.begin()+TIME_WINDOW;
    auto itDelete = data.begin();

    int resultCount = 0;

    uint64_t searchCycle = 0;
    uint64_t insertCycle = 0;
    uint64_t deleteCycle = 0;
    data[]

    for data:
        insert 
        search
        delete

        rebuild alex

    f
    for (int i = 0; i < NO_INSERT; ++i)
    {
        uint64_t tempInsertCycles = 0;
        startTimer(&tempInsertCycles);
        alex.insert(*it,i);
        stopTimer(&tempInsertCycles);
        insertCycle += tempInsertCycles;

        ++it;
    }

    for (int i = 0; i < NO_SEARCH; ++i)
    {
        uint64_t query_key = data.at((itDelete - data.begin()) + (rand() % ( (it - data.begin()) - (itDelete - data.begin()) + 1 )));
        
        uint64_t tempSearchCycles = 0;
        startTimer(&tempSearchCycles);
        // alex.lower_bound(query_key);
        auto it = alex.find(query_key);
        stopTimer(&tempSearchCycles);
        searchCycle += tempSearchCycles;

        resultCount = (it != alex.end())? resultCount + 1: resultCount;
    }

    for (int i = 0; i < NO_DELETE; ++i)
    {
        uint64_t tempDeleteCycles = 0;
        startTimer(&tempDeleteCycles);
        alex.erase(*itDelete);
        stopTimer(&tempDeleteCycles);
        deleteCycle += tempDeleteCycles;
        ++itDelete;
    }

    uint64_t totalCycle = searchCycle + insertCycle + deleteCycle;

    //Output into File
    ofstream run_time_out("./runtime_result.txt");
    assert(run_time_out);
    run_time_out << (double)totalCycle/MACHINE_FREQUENCY << endl;
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