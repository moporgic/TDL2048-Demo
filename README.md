# TDL2048-Demo

***Temporal Difference Learning for the Game of 2048 (Demo)***

[TDL2048-Demo](https://github.com/moporgic/TDL2048-Demo) is a foundational framework designed to explore *temporal difference (TD) learning* in the game of *2048*, which is built around the core principles of the *TD(0)* algorithm, incorporating an *n-tuple network* architecture that utilizes isomorphic features. Additionally, a *bitboard* representation is employed to manage game states and operations efficiently.

The demo is intended for beginners and enthusiasts looking to dive into reinforcement learning within the context of computer games. Whether you're aiming to understand TD learning mechanics, experiment with n-tuple networks, or investigate game AI strategies for 2048, TDL2048-Demo offers a solid starting point.

## Features

- **TD(0) Learning**: An implementation of the TD learning algorithm for 2048.
- **N-Tuple Network**: A network structure utilizes isomorphic puzzle patterns for 2048.
- **Bitboard**: A compact 64-bit bitboard representation for 2048.

## Quick Start

The demo is available in C++ ([`2048.cpp`](2048.cpp)) and Python ([`2048.py`](2048.py)), which are implemented with an identical design.

```bash
# for C++
make && ./2048
# for Python
python 2048.py
```

## Quick Links

* [Computer Games and Intelligence (CGI) Lab](https://cgi.lab.nycu.edu.tw)
* [Reinforcement Learning and Games (RLG) Lab](https://rlg.iis.sinica.edu.tw)

## References

1. M. Szubert and W. Jaśkowski, "Temporal difference learning of N-tuple networks for the game 2048," CIG 2014, doi: [10.1109/CIG.2014.6932907](https://doi.org/10.1109/CIG.2014.6932907).
2. I-C. Wu, K.-H. Yeh, C.-C. Liang, C.-C. Chang, and H. Chiang, "Multi-stage temporal difference learning for 2048," TAAI 2014, doi: [10.1007/978-3-319-13987-6_34](https://doi.org/10.1007/978-3-319-13987-6_34).
3. K. Matsuzaki, "Systematic selection of N-tuple networks with consideration of interinfluence for game 2048," TAAI 2016, doi: [10.1109/TAAI.2016.7880154](https://doi.org/10.1109/TAAI.2016.7880154).
