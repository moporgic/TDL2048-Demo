/**
 * TDL2048-Demo
 * Temporal Difference Learning Demo for Game 2x2 2048 (Demo)
 * Learning TD(0) by a 2x2 2048 after-state agent.
 *
 * use 'g++ -std=c++0x -O3 -g -o 2048 2048.cpp' to compile the source
 * https://github.com/moporgic/TDL2048-Demo
 *
 * Hung Guei
 * Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
 * http://www.aigames.nctu.edu.tw
 */
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <functional>
#include <numeric>

class board {
public:
	board() : tile({}) {}
	board(const board& b) : tile(b.tile) {}
	board(int v) : board() { operator =(v); }
	std::array<int, 2>& operator[](int i) { return tile[i]; }
	const std::array<int, 2>& operator[](int i) const { return tile[i]; }
	operator int() const {
		return tile[0][0] * 4096 + tile[0][1] * 256 + tile[1][0] * 16 + tile[1][1]; }
	board& operator=(int v) {
		tile[0][0] = (v >> 12) & 15;
		tile[0][1] = (v >> 8) & 15;
		tile[1][0] = (v >> 4) & 15;
		tile[1][1] = v & 15;
		return *this;
	}

	int move(const int& opcode) {
		switch (opcode) {
		case 0: return up();
		case 1: return right();
		case 2: return down();
		case 3: return left();
		default: return -1;
		}
	}

	int left() {
		int now = *this;
		int score = 0;
		for (auto& row : tile) {
			if (row[0] == 0) {
				row[0] = row[1];
				row[1] = 0;
			} else if (row[0] == row[1]) {
				row[0]++;
				row[1] = 0;
				score += (1 << row[0]) & -2u;
			}
		}
		return int(*this) != now ? score : -1;
	}
	int right() {
		mirror();
		int score = left();
		mirror();
		return score;
	}
	int up() {
		rotate(1);
		int score = right();
		rotate(-1);
		return score;
	}
	int down() {
		rotate(1);
		int score = left();
		rotate(-1);
		return score;
	}

	void transpose() {
		std::swap(tile[0][1], tile[1][0]);
	}

	void mirror() {
		std::swap(tile[0][0], tile[0][1]);
		std::swap(tile[1][0], tile[1][1]);
	}

	void flip() {
		std::swap(tile[0][0], tile[1][0]);
		std::swap(tile[0][1], tile[1][1]);
	}

	void rotate(const int& r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: transpose(); mirror(); break;
		case 2: mirror(); flip(); break;
		case 3: transpose(); flip(); break;
		}
	}

	void isomorphic(const int& i) {
		int iso = ((i % 8) + 8) % 8;
		if (iso > 4) mirror();
		rotate(iso);
	}

	void next() {
		int space[4], num = 0;
		if (tile[0][0] == 0) space[num++] = 0;
		if (tile[0][1] == 0) space[num++] = 1;
		if (tile[1][0] == 0) space[num++] = 2;
		if (tile[1][1] == 0) space[num++] = 3;
		if (num == 0) return;
		int pos = space[std::rand() % num];
		int pop = std::rand() % 10 ? 1 : 2;
		tile[pos / 2][pos % 2] = pop;
	}

	std::string name() const {
		std::stringstream ss;
		char buff[8];
		std::snprintf(buff, sizeof(buff), "%04x", int(*this));
		ss << buff;
		return ss.str();
	}

    friend std::ostream& operator <<(std::ostream& out, const board& b) {
		char buff[32];
		out << "+--------+" << std::endl;
		std::snprintf(buff, sizeof(buff), "|%4u%4u|", (1 << b[0][0]) & -2u, (1 << b[0][1]) & -2u);
		out << buff << std::endl;
		std::snprintf(buff, sizeof(buff), "|%4u%4u|", (1 << b[1][0]) & -2u, (1 << b[1][1]) & -2u);
		out << buff << std::endl;
		out << "+--------+" << std::endl;
		return out;
	}

private:
	std::array<std::array<int, 2>, 2> tile;
};



int main(int argc, const char* argv[]) {
	std::srand(std::time(nullptr));

	float V[65536] = { 0 };

	float alpha = 0.1;
	int decimal = 4;
	bool iso = true;

	auto norm = [=](const float& v) {
		double base = std::pow(10, decimal);
		return std::round(v * base) / base;
	};

	std::vector<board> history; history.reserve(100);
	std::vector<int> actions; actions.reserve(50);

	auto append_board_at = [&](std::string buff[4], int i) {
		board b = history[i];
		std::stringstream ss;
		ss << b;
		std::string line;
		for (int i = 0; i < 4 && std::getline(ss, line); i++)
			buff[i] += line.substr(1);
		if (i % 2) {
			std::string n_ = b.name();
			buff[3][buff[3].size() - 8] = '[';
			buff[3][buff[3].size() - 3] = ']';
			std::copy_n(n_.begin(), n_.size(), buff[3].begin() + (buff[3].size() - 7));

			b = history[i - 1];
			int r = b.move(actions[i / 2]);
			std::string r_ = std::to_string(r);
			buff[0][buff[0].size() - r_.size() - 5] = '(';
			buff[0][buff[0].size() - r_.size() - 4] = '+';
			buff[0][buff[0].size() - 3] = ')';
			std::copy_n(r_.begin(), r_.size(), buff[0].begin() + (buff[0].size() - r_.size() - 3));
		}
	};

	auto append_action_at = [&](std::string buff[4], int k) {
		board b = history[k - 1];
		board a[4] = { b, b, b, b };
		int r[4] = { -1 };
		for (int op = 0; op < 4; op++) {
			r[op] = a[op].move(op);
		}
		std::string opname[] = { "^", ">", "v", "<" };
		for (int i = 0; i < 4; i++) {
			std::cout << buff[i] << " " << opname[i] << ": " << r[i];
			if (r[i] != -1) {
				std::cout << " + " << norm(V[a[i]]);
				if (int(a[i]) == int(history[k])) std::cout << " *";
				std::cout << std::endl;
			} else {
				std::cout << std::endl;
			}
		}
	};

	for (size_t i = 1; true; i++) {
		bool print = true;
		bool bypass = false;

		if (print) std::cout << "episode #" << i << ":" << std::endl;
		board b;
		while (true) {
			b.next();
			history.push_back(b);
			int x = 0;
			board a[4] = { b, b, b, b };
			int r[4] = { -1 };
			float v[4] = { -std::numeric_limits<float>::max() };
			for (int op = 0; op < 4; op++) {
				r[op] = a[op].move(op);
				if (r[op] != -1) {
					v[op] = r[op] + V[a[op]];
					if (v[op] > v[x]) x = op;
				}
			}
			if (r[x] == -1) break;
			b = a[x];
			history.push_back(b);
			actions.push_back(x);
		}
		if (print) {
			for (size_t k = 1; k < history.size(); k += 2) {
				std::string buff[4] = { "+", "|", "|", "+" };
				for (size_t i = 0; i < k; i++) {
					append_board_at(buff, i);
				}
				append_action_at(buff, k);
			}
			std::string buff[4] = { "+", "|", "|", "+" };
			for (size_t i = 0; i < history.size(); i++) {
				append_board_at(buff, i);
			}
			for (std::string& line : buff) {
				std::cout << line << std::endl;
			}
		}

		float exact = 0;
		history.pop_back();
		while (history.size()) {
			auto& v = V[history.back()];
			auto upd = alpha * (exact - v);
			if (print) {
				std::cout << "v[" << history.back().name() << "] is " << norm(v);
				std::cout << ", should be " << norm(exact) << ": ";
				if (norm(v) != norm(exact)) {
					std::cout << "v[" << history.back().name() << "] = ";
					std::cout << norm(v) << " + " << alpha << " * (" << norm(exact) << " - " << norm(v) << ") = ";
					std::cout << norm(v + upd) << std::endl;
				} else {
					std::cout << "correct" << std::endl;
				}
			}
			v += upd;
			if (iso) {
				std::vector<int> trained;
				trained.reserve(8);
				board a = history.back();
				trained.push_back(a);
				for (int i = 1; i < 8; i++) {
					board iso = a;
					iso.isomorphic(i);
					if (std::find(trained.begin(), trained.end(), iso) == trained.end()) {
						trained.push_back(iso);
						V[iso] += upd;
					}
				}
			}
			exact = v;
			history.pop_back();
			exact += board(history.back()).move(actions.back());
			history.pop_back();
			actions.pop_back();
		}

		if (!bypass) std::getchar();
	}

	return 0;
}
