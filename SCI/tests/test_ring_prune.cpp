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
int bw_x = 32;
int bw_y = 32;
int s_x = 12;
int s_y = 12;
int num_prune = 0;

int num_heads = 16;

uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];

double computeULPErr(double calc, double actual) {
  return (calc>actual)? (calc - actual) : (actual - calc);
}

// todo, x should be 2D array
void prune_thread(int tid, uint64_t *x, uint64_t *score, uint64_t *mask, uint64_t *y, int num_ops) {
  FixOp *fix;
  if (tid & 1) {
    fix = new FixOp(3 - party, iopackArr[tid], otpackArr[tid]);
  } else {
    fix = new FixOp(party, iopackArr[tid], otpackArr[tid]);
  }
    FixArray token_fixed_array = fix->input((tid&1)?(3-party):(party), dim, x, false, bw_x, s_x);
    FixArray score_fixed_array = fix->input((tid&1)?(3-party):(party), dim, score, false, bw_x, s_x);
    // learn the mask
    // FixArray mask_array = fix->gen_mask(score_fixed_array, bw_y, s_y);
    // memcpy(mask, mask_array.data, dim * sizeof(uint64_t));
    FixArray y_fixed_array = fix->prune(token_fixed_array, score_fixed_array, bw_y, s_y, num_prune);
    // memcpy(y, y_fixed_array.data, dim * sizeof(uint64_t));
    memcpy(y, y_fixed_array.data, dim * sizeof(uint64_t));
    delete fix;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of prune operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("pr", num_prune, "number of tokens to prune");

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
  // for this test, the input is 1 token sequence of [N, d] and H attention maps of [n, n]
  // the output is a pruned token sequence of [N', d], where the #token is decreased
  
  // H heads, each has a N*N (dim*dim) attention map
  std::vector<uint64_t *> att_map;
  for(int i=0; i<num_heads; i++){
    att_map.push_back(new uint64_t[dim*dim]);
  }

  for(int h=0; h<num_heads; h++){
    prg.random_data(att_map[h], dim*dim * sizeof(uint64_t));
      for(int i = 0; i <dim*dim; i++){
        att_map[h][i] &= uint64_t(0x3fff);
      }
  }

   cout <<BLUE << "[Multihead] " << num_heads << " heads and N=" <<  dim << RESET << endl;

  uint64_t *x = new uint64_t[dim];
  // std::vector<std::vector<uint64_t>> x(dim, std::vector<uint64_t>(1));
  uint64_t *mask = new uint64_t[dim];
  uint64_t *y = new uint64_t[dim];
  uint64_t data[10] = {0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x7000,0x8000,0x9000,0xa000};
  for(int i = 0; i <dim; i++){
      x[i] = data[i%10]>>1;
  }


  // make x itself small, so sub won't overflow
  for (int i = 0; i < dim; i++) {
    x[i] &= uint64_t(0xffff);
  }

  // compute score (on sharing) locally
  // score[i] = sum_head, sum_col Att / H*col
  uint64_t *score = new uint64_t[dim];
  // uint64_t importance[10] = {0x1000,0x6000,0x2000,0x7000,0x3000,0x8000,0x4000,0x9000,0x5000,0xa000};
  uint64_t importance[10] = {0xa000,0xa000,0xa000,0xa000,0xa000,0xa000,0xa000,0xa000,0xa000,0xa000};
  // prg.random_data(score, dim * sizeof(uint64_t));

  for(int i = 0; i <dim; i++){
      score[i] = importance[i%10]>>1;
      score[i] &= uint64_t(0xffff);
  }

  // set the number to be pruned
  int prune_num = num_prune ;
  for(int i = 0; i < prune_num; i++){
      score[i] = (0x1000)>>1;
      score[i] &= uint64_t(0xffff);
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
        std::thread(prune_thread, i, x + offset, score + offset, mask + offset, y + offset, lnum_ops);
        // std::thread(prune_thread, i, score + offset, y + offset, lnum_ops);
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
    iopackArr[0]->io->send_data(score, dim * sizeof(uint64_t));
    iopackArr[0]->io->send_data(mask, dim * sizeof(uint64_t));
  } else { // party == BOB
    // y0 is Alice's share of result
    uint64_t *x0 = new uint64_t[dim];
    uint64_t *y0 = new uint64_t[dim];
    uint64_t *score0 = new uint64_t[dim];
    uint64_t *mask0 = new uint64_t[dim];
    iopackArr[0]->io->recv_data(x0, dim * sizeof(uint64_t));
    iopackArr[0]->io->recv_data(y0, dim * sizeof(uint64_t));
    iopackArr[0]->io->recv_data(score0, dim * sizeof(uint64_t));
    iopackArr[0]->io->recv_data(mask0, dim * sizeof(uint64_t));


    double *x_full = new double[dim];
    double *y_full = new double[dim];
    double *score_full = new double[dim];
    double *mask_alice = new double[dim];
    double *mask_bob = new double[dim];
    double *actual_mask = new double[dim];
    int success=1;
    for (int i = 0; i < dim; i++) {
      score_full[i] = (signed_val(score0[i] + score[i], bw_x)) / double(1LL << s_x);
      x_full[i] = (signed_val(x0[i] + x[i], bw_x)) / double(1LL << s_x);
      y_full[i] = (signed_val((y0[i] + y[i])&uint64_t(0x0ffff), bw_x)) / double(1LL << s_x);
      mask_alice[i] =  (uint64_t(mask0[i]))&1;
      mask_bob[i] = (uint64_t(mask[i]))&1;
      actual_mask[i] = (score_full[i] > 5.0) ? 1 : 0;
    }

    if(success == 1){
      cout << GREEN << "[Prune] Success mask! " << RESET<< endl;
    }
    success = 1;
    // if all tokens are correctly swapped
    size_t j = 0; // Pointer for position to swap to
    for (size_t i = 0; i < dim; ++i) {
        if (score_full[i] > 5) {
            if (i != j) {
                std::swap(x_full[i], x_full[j]); // Swap x values
            }
            j++;
        }
    }

    cout << "================== res and shares ==================\n";
    cout << "prune " << num_prune << " out of "  << dim << "\n";

    cout << "real x = \t";
    for(int32_t i =0; i< dim; i++){
        double db_tmp = (signed_val(x0[i] + x[i], bw_x)) / double(1LL << s_x);
        cout << db_tmp <<"  ";
    }
    cout<<endl;

    cout << "real score = \t";
    for(int32_t i =0; i< dim; i++){
        double db_tmp = (signed_val(score0[i] + score[i], bw_x)) / double(1LL << s_x);
        cout << db_tmp <<"  ";
    }
    cout<<endl;

    // doing prune needs bw_y-=3, but prune don't
    cout << "real pruned =\t";
    for(int32_t i =0; i< dim; i++){
        double db_tmp = x_full[i];
        cout << db_tmp <<"  ";
    }
    cout<<endl;

    cout << "added pruned =\t";
    for(int32_t i =0; i< dim; i++){
        double db_tmp = y_full[i];
        cout << db_tmp <<"  ";
    }
    cout<<endl;


    // evaluate compute error
    double total_err = 0;
    double max_ULP_err = 0;

    cerr << "Average ULP error: " << total_err / dim << endl;
    cerr << "Max ULP error: " << max_ULP_err << endl;
    cerr << "Number of tests: " << dim << endl;

    delete[] x0;
    delete[] y0;

    delete[] score_full;
    delete[] mask_alice;
    delete[] mask_bob;
    delete[] actual_mask;
  }

  cout << "Number of prune ops/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  cout << "prune Time\t" << t / (1000.0) << " ms" << endl;
  cout << "prune Bytes Sent\t" << total_comm << " bytes" << endl;

  if (party == ALICE){
      cout<< BLUE << "[Info] Called SCI_OT for Prune"<< RESET << endl;
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
