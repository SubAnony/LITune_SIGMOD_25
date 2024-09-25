test: index_test.cpp
	g++ index_test.cpp -std=c++17 -march=native -O3 -w -o z_run_test

execute: z_run_test
	./z_run_test

compile_and_execute: index_test.cpp
	g++ index_test.cpp -std=c++17 -march=native -O3 -w -o z_run_test && ./z_run_test

run_benchmark: run_carmi.cpp
	g++ run_carmi.cpp -std=c++17 -march=native -O3 -w -o z_run_test && ./z_run_test

clean:
	rm *.o z_run_test

clean_benchmark_output:
	rm -f runtime_result.txt state_result.txt