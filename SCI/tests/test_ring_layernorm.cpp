#include "Math/math-functions.h"
#include "FloatingPoint/fixed-point.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>

using namespace sci;
using namespace std;

#define MAX_THREADS 4

int party, port = 32000;
// int num_threads = 4;
int num_threads = 1;
string address = "127.0.0.1";

int dim = 10000;
// int bw_x = 17;
// int bw_y = 17;
int bw_x = 32;
int bw_y = 32;
int s_x = 12;
int s_y = 12;

uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));
uint64_t mask_value = (bw_x == 64 ? -1 : ((1ULL << (bw_x - 3)) - 1));

IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];

double computeULPErr(double calc, double actual) {
  return (calc>actual)? (calc - actual) : (actual - calc);
}



// math-function version
void layernorm_thread(int tid, uint64_t *x, uint64_t *y, int num_ops) {
  MathFunctions *math;
  if (tid & 1) {
    math = new MathFunctions(3 - party, iopackArr[tid], otpackArr[tid]);
  } else {
    math = new MathFunctions(party, iopackArr[tid], otpackArr[tid]);
  }
    math->layer_norm(num_ops, x, y, bw_x, bw_y, s_x, s_y);
    delete math;
}


void layernorm(int n, double *x, double *y){
  //take mean, sqrt_avg_((x-mean)^2)
  double sum_x=0, mean=0;
  for(int i=0; i<n; i++){
    sum_x += x[i];
  }
  mean = sum_x / (n*1.0);
  double sum_t=0, var_2=0, var=0;
  for(int i=0; i<n; i++){
    sum_t += (x[i] - mean) * (x[i] - mean);
  }
  var_2 = sum_t / (n*1.0);
  var = sqrt(var_2);
  for(int i=0; i<n; i++){
    y[i] = (x[i] - mean) / var;
  }
  return;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of layernorm operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);

  assert(num_threads <= MAX_THREADS);

  /********** Setup IO and Base OTs ***********/
  /********************************************/
  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }
  std::cout << "All Base OTs Done" << std::endl;

  /************ Generate Test Data ************/
  /********************************************/
  PRG128 prg;

  uint64_t *x = new uint64_t[dim];
  uint64_t *y = new uint64_t[dim];

  // prg.random_data(x, dim * sizeof(uint64_t));

  // uint64_t data[10] = {0x1000,0x1000,0x1000,0x1000,0x1000,0x2000,0x2000,0x2000,0x2000,0x3000};
  // for(int i = 0; i <10; i++){
  //   x[i] = data[i];
  // } 

  uint64_t data[3] = {0x1111,0x2111,0x3111};
  for(int i = 0; i <dim; i++){
      x[i] = data[i%3];
  } 

  // make x itself small, so sub won't overflow
  // cout<< std::hex<<mask_tmp<<std::dec<<endl;
  for (int i = 0; i < dim; i++) {
    x[i] &= mask_value;
  }

  /************** Fork Threads ****************/
  /********************************************/
  uint64_t total_comm = 0;
  uint64_t thread_comm[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm();
  }

  // start timing
  auto start = clock_start();
  std::thread sig_threads[num_threads];
  int chunk_size = dim / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_ops;
    if (i == (num_threads - 1)) {
      lnum_ops = dim - offset;
    } else {
      lnum_ops = chunk_size;
    }
    sig_threads[i] =
        std::thread(layernorm_thread, i, x + offset, y + offset, lnum_ops);
  }
  for (int i = 0; i < num_threads; ++i) {
    sig_threads[i].join();
  }
  // end timing
  long long t = time_from(start);

  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm() - thread_comm[i];
    total_comm += thread_comm[i];
  }

  /************** Verification ****************/
  /********************************************/
  if (party == ALICE) {
    iopackArr[0]->io->send_data(x, dim * sizeof(uint64_t));
    iopackArr[0]->io->send_data(y, dim * sizeof(uint64_t));
  } else { // party == BOB
    // y0 is Alice's share of result
    uint64_t *x0 = new uint64_t[dim];
    uint64_t *y0 = new uint64_t[dim];
    iopackArr[0]->io->recv_data(x0, dim * sizeof(uint64_t));
    iopackArr[0]->io->recv_data(y0, dim * sizeof(uint64_t));


    double total_err = 0;
    double max_ULP_err = 0;
    double *x_full = new double[dim];
    double *y_full = new double[dim];
    double *y_actual = new double[dim];
    for (int i = 0; i < dim; i++) {
      x_full[i] = (signed_val(x0[i] + x[i], bw_x-15)) / double(1LL << s_x);
      y_full[i] = (signed_val( y0[i] + y[i], bw_y-15)) / double(1ULL << s_y);
      layernorm(dim, x_full, y_actual);
    }


    for (int i = 0; i < dim; i++) {
      double err = computeULPErr(y_full[i], y_actual[i]);
      total_err += err;
      max_ULP_err = std::max(max_ULP_err, err);
    }

    cerr << "Average ULP error: " << total_err / dim << endl;
    cerr << "Max ULP error: " << max_ULP_err << endl;
    cerr << "Number of tests: " << dim << endl;

    delete[] x0;
    delete[] y0;
    delete[] x_full;
    delete[] y_full;
    delete[] y_actual;
  }

  cout << "Number of layernorm ops/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  cout << "layernorm Time\t" << t / (1000.0) << " ms" << endl;
  cout << "layernorm Bytes Sent\t" << total_comm << " bytes" << endl;

  if (party == ALICE){
      cout<< BLUE << "[Info] Called SCI_OT for LayerNorm"<< RESET << endl;
      cout<< BLUE << "[Info] Vector Length is: " << dim << RESET << endl;
      cout << GREEN << "[Server] Successful Operation" << RESET << endl;
  }


  /******************* Cleanup ****************/
  /********************************************/
  delete[] x;
  delete[] y;
  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }
}