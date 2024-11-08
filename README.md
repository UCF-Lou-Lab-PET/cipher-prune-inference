# CipherPrune: Efficient and Scalable Private Transformer Inference

This is the code for CipherPrune: Efficient and Scalable Private Transformer Inference. This code is built upon the [EzPC](https://github.com/mpc-msri/EzPC) library.

## Installation

Please follow the [EzPC](https://github.com/mpc-msri/EzPC) instructions to build and install the library.

## Test pruning

To test the secure pruning protocol, run the following command under the ```SCI/build/bin``` directoty. One token is a vector of length 768 by default. For example, to prune 4 tokens out of 55 tokens, run (role==1 means server).

    ./prune-OT r=1 N=50 pr=2

In another terminal run (role==1 means client):

    ./prune-OT r=2 N=50 pr=2

By successfully building the test files, on the server side you should see:

    connected
    All Base OTs Done
    [Multihead] 16 heads and N=50
    [Test] in prune
    [In prune] Time taken: 0.918475 seconds
    [In Prune] prune Time	528.661 ms
    Number of prune ops/s:	94.4415
    prune Time	529.428 ms
    prune Bytes Sent	3031806 bytes
    [Info] Called SCI_OT for Prune
    [Info] Vector Length is: 50
    [Server] Successful Operation

and on the client side, you should see:

    connected
    All Base OTs Done
    [Multihead] 16 heads and N=50
    [Test] in prune
    [In prune] Time taken: 0.900247 seconds
    [In Prune] prune Time	528.909 ms
    [Prune] Success mask!
    ================== res and shares ==================
    prune 2 out of 50
    Average ULP error: 0
    Max ULP error: 0
    Number of tests: 50
    Number of prune ops/s:	94.4615
    prune Time	529.316 ms
    prune Bytes Sent	4624846 bytes