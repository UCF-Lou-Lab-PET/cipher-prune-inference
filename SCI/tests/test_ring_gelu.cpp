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

void gelu_thread(int tid, uint64_t *x, uint64_t *y, int num_ops) {
  FixOp *fix;
  if (tid & 1) {
    fix = new FixOp(3 - party, iopackArr[tid], otpackArr[tid]);
  } else {
    fix = new FixOp(party, iopackArr[tid], otpackArr[tid]);
  }
    FixArray x_fixed_array = fix->input((tid&1)?(3-party):(party), dim, x, 1, bw_x, s_x);
    FixArray y_fixed_array = fix->GeLU(x_fixed_array, bw_y, s_y);
    memcpy(y, y_fixed_array.data, dim * sizeof(uint64_t));
    delete fix;
}

double gelu(double x){
    const double c = sqrt(2.0 / M_PI) * (1.0 + 0.044715 * pow(x, 3));
    return 0.5 * x * (1.0 + tanh(c));
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of GeLU operations");
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

  uint64_t data[3] = {0x1111,0x2111,0x3111};
  prg.random_data(x, dim * sizeof(uint64_t));
  for(int i = 0; i <dim; i++){
      x[i] &= uint64_t(0x3fff);
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
        std::thread(gelu_thread, i, x + offset, y + offset, lnum_ops);
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


    // evaluate compute error
    double total_err = 0;
    double max_ULP_err = 0;
    double *x_full = new double[dim];
    double *y_full = new double[dim];
    double *y_actual = new double[dim];
    for (int i = 0; i < dim; i++) {
      x_full[i] = (signed_val(x0[i] + x[i], bw_x)) / double(1LL << s_x);
      y_full[i] = (signed_val( y0[i] + y[i], bw_y-3)) / double(1ULL << s_y);
      y_actual[i] = gelu(x_full[i]);
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

  cout << "Number of GeLU ops/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  cout << "GeLU Time\t" << t / (1000.0) << " ms" << endl;
  cout << "GeLU Bytes Sent\t" << total_comm << " bytes" << endl;

  if (party == ALICE){
      cout<< BLUE << "[Info] Called SCI_OT for GuLE"<< RESET << endl;
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
