#Writing to the parameter file

#Read from txt file  
#...

#Write to parameters
> src/parameters.hpp;
echo "#pragma once" >> src/parameters.hpp;
echo "#define external_kMaxLeafNodeSize 256" >> src/parameters.hpp;
echo "#define external_kMaxLeafNodeSizeExternal 1024" >> src/parameters.hpp;
echo "#define external_kAlgorithmThreshold 60000" >> src/parameters.hpp;
echo "#define external_kMemoryAccessTime 80.09" >> src/parameters.hpp;
echo "#define external_kLRRootTime 11.54" >> src/parameters.hpp;
echo "#define external_kPLRRootTime 29.62" >> src/parameters.hpp;
echo "#define external_kLRInnerTime kMemoryAccessTime + 5.23" >> src/parameters.hpp;
echo "#define external_kPLRInnerTime kMemoryAccessTime + 22.8" >> src/parameters.hpp;
echo "#define external_kHisInnerTime kMemoryAccessTime + 18.44" >> src/parameters.hpp;
echo "#define external_kBSInnerTime kMemoryAccessTime + 26.38" >> src/parameters.hpp;
echo "#define external_kCostMoveTime 6.25" >> src/parameters.hpp; 
echo "#define external_kLeafBaseTime kMemoryAccessTime + 25.4" >> src/parameters.hpp; 
echo "#define external_kCostBSTime 10.9438" >> src/parameters.hpp; 
echo "#define external_lambda -1" >> src.parameters.hpp;

#Compile
#...