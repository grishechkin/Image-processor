#include "iostream"
#include "omp.h"
#include "fstream"
#include "string"
#include "iomanip"
#include "cassert"

#define calc_q(left, right) (prefsum[right + 1] - prefsum[left])
#define calc_m(left, right) (prefsum_m[right + 1] - prefsum_m[left])

using namespace std;

const short int L = 255;
const short int M = 4;
const uint8_t C0 = 0;
const uint8_t C1 = 84;
const uint8_t C2 = 170;
const uint8_t C3 = 255;

size_t count_of_elements;
size_t prefsum[L + 2];
size_t prefsum_m[L + 2];

void solution_without_parallel(uint8_t* data, int begin_pos, int size) {
    long long gist[L + 1];
    for (long long & i : gist) {
        i = 0;
    }
    for (int i = begin_pos; i < size; i++) {
        gist[data[i]]++;
    }

    for (int i = 0; i < L + 1; i++) {
        prefsum[i + 1] = prefsum[i] + gist[i];
        prefsum_m[i + 1] = prefsum_m[i] + gist[i] * i;
    }

    double max_delta = 0;
    int ans[3];
    double delta;
    size_t m1;
    size_t m2;
    size_t m3;
    size_t m4;
    for (short int f0 = 0; f0 < L - M; f0++) {
        for (short int f1 = f0 + 1; f1 < (short int) (L - M + 1); f1++) {
            for (short int f2 = f1 + 1; f2 < (short int) (L - M + 2); f2++) {
                m1 = calc_m(0, f0);
                m2 = calc_m(f0 + 1, f1);
                m3 = calc_m(f1 + 1, f2);
                m4 = calc_m(f2 + 1, L);

                delta = (double) (m1 * m1) / calc_q(0, f0);
                delta += (double) (m2 * m2) / calc_q(f0 + 1, f1);
                delta += (double) (m3 * m3) / calc_q(f1 + 1, f2);
                delta += (double) (m4 * m4) / calc_q(f2 + 1, L);

                if (delta > max_delta) {
                    max_delta = delta;
                    ans[0] = f0;
                    ans[1] = f1;
                    ans[2] = f2;
                }
            }
        }
    }

    for (int i = begin_pos; i < size; i++) {
        if (data[i] < ans[0]) data[i] = C0;
        else if (data[i] < ans[1]) data[i] = C1;
        else if (data[i] < ans[2]) data[i] = C2;
        else data[i] = C3;  
    }

    cout << ans[0] << " " << ans[1] << " " << ans[2] << "\n";
}

void solution_with_parallel(uint8_t* data, int begin_pos, int size) {
    long long gist[L + 1];

#pragma omp parallel
    {
#pragma omp for
        for (short int i = 0; i < L + 1; i++) {
            gist[i] = 0;
        }
        long long thread_gist[L + 1];
        for (short int i = 0; i < L + 1; i++) {
            thread_gist[i] = 0;
        }

#pragma omp for nowait
        for (int i = begin_pos; i < size; i++) {
            thread_gist[data[i]]++;
        }

#pragma omp critical
        {
            for (short int i = 0; i < L + 1; i++) {
                gist[i] += thread_gist[i];
            }
        };
    };

    for (short int i = 0; i < L + 1; i++) {
        prefsum[i + 1] = prefsum[i] + gist[i];
        prefsum_m[i + 1] = prefsum_m[i] + gist[i] * i;
    }

    double max_delta = 0;
    short int ans[3];
    #pragma omp parallel
    {
        double thread_max_delta = 0;
        short int thread_ans[3];
        double delta;
        size_t m1;
        size_t m2;
        size_t m3;
        size_t m4;
        for (short int f0 = 0; f0 < L - M; f0++) {
            for (short int f1 = f0 + 1; f1 < (short int) (L - M + 1); f1++) {
    #pragma omp for nowait
                for (short int f2 = f1 + 1; f2 < (short int) (L - M + 2); f2++) {
                    m1 = calc_m(0, f0);
                    m2 = calc_m(f0 + 1, f1);
                    m3 = calc_m(f1 + 1, f2);
                    m4 = calc_m(f2 + 1, L);

                    delta = (double) (m1 * m1) / calc_q(0, f0);
                    delta += (double) (m2 * m2) / calc_q(f0 + 1, f1);
                    delta += (double) (m3 * m3) / calc_q(f1 + 1, f2);
                    delta += (double) (m4 * m4) / calc_q(f2 + 1, L);

                    if (delta > thread_max_delta) {
                        thread_max_delta = delta;
                        thread_ans[0] = f0;
                        thread_ans[1] = f1;
                        thread_ans[2] = f2;
                    }
                }
            }
        }
    #pragma omp critical
        {
            if (max_delta < thread_max_delta) {
                max_delta = thread_max_delta;
                ans[0] = thread_ans[0];
                ans[1] = thread_ans[1];
                ans[2] = thread_ans[2];
            }
        };
    }

#pragma omp parallel
    {
        #pragma omp for
        for (int i = begin_pos; i < size; i++) {
            if (data[i] < ans[0]) data[i] = C0;
            else if (data[i] < ans[1]) data[i] = C1;
            else if (data[i] < ans[2]) data[i] = C2;
            else data[i] = C3;
        }
    }

    cout << ans[0] << " " << ans[1] << " " << ans[2] << "\n";
}

int main(int number_of_arguments, char *args[]) {
    assert("Incorrect number of arguments" && number_of_arguments == 4);
    int number_of_threads;
    try {
        number_of_threads = stoi(args[1]);
    } catch (...) {
        cerr << "Incorrect arguments";
        return 0;
    }
    assert("Incorrect number of threads" && number_of_threads >= -1);

    cout << fixed << setprecision(5);

    try {
        ifstream input_file(args[2], ios_base::binary);
        input_file.exceptions(ifstream::failbit);

        try {
            input_file.seekg(0, ifstream::end);
            long long size = input_file.tellg();
            input_file.seekg(0, ifstream::beg);

            auto *data = new uint8_t[size];
            try {
                input_file.read((char*) data, size);

                string p5;
                p5 += data[0];
                p5 += data[1];
                p5 += data[2];
                assert(p5 == "P5\n");

                string w, h;
                int pos = 3;
                while (data[pos] != ' ') w += data[pos++];
                pos++;
                while (data[pos] != '\n') h += data[pos++];

                try {
                    count_of_elements = stoi(w) * stoi(h);
                } catch (...) {
                    cerr << "Incorrect data";
                    input_file.close();
                    return 0;
                }

                pos++;
                string s255;
                s255 += data[pos++];
                s255 += data[pos++];
                s255 += data[pos++];
                s255 += data[pos++];
                assert(s255 == "255\n");
                assert(pos + count_of_elements == size);

                double begin_time = omp_get_wtime();
                if (number_of_threads == -1) {
                    solution_without_parallel(data, pos, size);
                } else {
                    if (number_of_threads != 0) omp_set_num_threads(number_of_threads);
                    solution_with_parallel(data, pos, size);  
                }
                printf("Time (%i thread(s)): %g ms\n", number_of_threads, (omp_get_wtime() - begin_time) * 1000);
            
                ofstream output_file(args[3], ios_base::binary);
                try {
                    output_file.write((char*) data, size);    
                } catch (...) {
                    output_file.close();
                    input_file.close();
                    delete[] data;  
                    cerr << "Error";
                    return 0;
                }
                
                output_file.close();
                input_file.close();
                delete[] data;   
            } catch (...) {
                input_file.close();
                delete[] data;
                cerr << "Error";
                return 0;
            }
        } catch (...) {
            input_file.close();
            cerr << "Error";
            return 0;
        }
    } catch (...) {
        cout << "Error" << endl;
        return 0;
    }
    
    return 0;
}
