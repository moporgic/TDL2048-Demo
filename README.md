# TDL2048-Demo

***Temporal Difference Learning for the Game of 2048 (Demo)***

[TDL2048-Demo](https://github.com/moporgic/TDL2048-Demo) is a foundational framework designed to explore *temporal difference (TD) learning* in the game of *2048*, which is built around the core principles of the *TD(0)* algorithm, incorporating an *n-tuple network* architecture that utilizes isomorphic features, and employing a *bitboard* representation to manage game states and operations efficiently.

The demo is intended for beginners and enthusiasts looking to dive into reinforcement learning within the context of computer games. Whether you're aiming to understand TD learning mechanics, experiment with n-tuple networks, or investigate game AI strategies for 2048, TDL2048-Demo offers a solid starting point.

## Features at a Glance

- **TD(0) Learning**: An implementation of the TD learning algorithm for 2048.
- **N-Tuple Network**: A network structure utilizes isomorphic puzzle patterns for 2048.
- **Bitboard**: A compact 64-bit bitboard representation for 2048.

## Quick Start Guide

The ready-to-run examples are available in C++ ([2048.cpp](2048.cpp)) and Python ([2048.py](2048.py)), which are implemented using an identical design.

```bash
# For C++
make && ./2048

# For Python
python 2048.py
```

The program prints the statistics every 1000 games, which include average score, maximum score, and tile distribution.
The example below is the statistics from the 99001st to the 100000th games.
```
100000  avg = 68663.7   max = 177508
        256     100%    (0.2%)
        512     99.8%   (0.9%)
        1024    98.9%   (7.7%)
        2048    91.2%   (22.5%)
        4096    68.7%   (53.9%)
        8192    14.8%   (14.8%)
```
- `100000`: the current iteration, i.e., number of games trained.
- `avg = 68663.7  max = 177508`: the average score is 68663.7;
                                 the maximum score is 177508.
- `2048  91.2%  (22.5%)`: 91.2% of games reached 2048-tiles, i.e., the win rate of 2048-tile;
                          22.5% of games terminated when 2048-tiles were the largest tile.

## Collaborating Institutions

* [Computer Games and Intelligence (CGI) Lab](https://cgi.lab.nycu.edu.tw), Department of Computer Science, National Yang Ming Chiao Tung University, Taiwan
* [Reinforcement Learning and Games (RLG) Lab](https://rlg.iis.sinica.edu.tw), Institute of Information Science, Academia Sinica, Taiwan

## References

1. M. Szubert and W. Jaśkowski, "Temporal difference learning of N-tuple networks for the game 2048," CIG 2014, doi: [10.1109/CIG.2014.6932907](https://doi.org/10.1109/CIG.2014.6932907).
2. I-C. Wu, K.-H. Yeh, C.-C. Liang, C.-C. Chang, and H. Chiang, "Multi-stage temporal difference learning for 2048," TAAI 2014, doi: [10.1007/978-3-319-13987-6_34](https://doi.org/10.1007/978-3-319-13987-6_34).
3. K. Matsuzaki, "Systematic selection of N-tuple networks with consideration of interinfluence for game 2048," TAAI 2016, doi: [10.1109/TAAI.2016.7880154](https://doi.org/10.1109/TAAI.2016.7880154).
