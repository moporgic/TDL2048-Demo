/**
 * Temporal Difference Learning Demo for Game 2048
 * use 'g++ -O3 -o 2048 2048.cpp' to compile the source
 *
 * Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
 * http://www.aigames.nctu.edu.tw/
 * January 2017
 *
 * References:
 * [1] Szubert, Marcin, and Wojciech Jaśkowski. "Temporal difference learning of n-tuple networks for the game 2048."
 * Computational Intelligence and Games (CIG), 2014 IEEE Conference on. IEEE, 2014.
 * [2] Wu, I-Chen, et al. "Multi-stage temporal difference learning for 2048."
 * Technologies and Applications of Artificial Intelligence. Springer International Publishing, 2014. 366-378.
 */
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdarg>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

/**
 * output stream wrapper
 */
class output {
private:
	std::ostream& out;
	bool enable;
public:
	output(std::ostream& out, const bool& en = true) : out(out), enable(en) {}
	template<typename type>
	inline output& operator <<(const type& v) {
		if (enable) out << v;
		return *this;
	}
	inline output& operator <<(std::ostream& (*pf)(std::ostream&)) {
		if (enable) out << pf;
		return *this;
	}
} info(std::cout, true), error(std::cerr, true), debug(std::cout, false);

/**
 * The simplest bitboard implementation for 2048 board
 */
class board {
public:
	typedef unsigned long long value_t;

	inline board(const value_t& raw = 0) : raw(raw) {}
	inline board(const board& b) : raw(b.raw) {}
	inline operator value_t&() { return raw; }

	inline int  fetch(const int& i) const { return ((raw >> (i << 4)) & 0xffff); }
	inline void place(const int& i, const int& r) { raw = (raw & ~(0xffffULL << (i << 4))) | (value_t(r & 0xffff) << (i << 4)); }
	inline int  at(const int& i) const { return (raw >> (i << 2)) & 0x0f; }
	inline void set(const int& i, const int& t) { raw = (raw & ~(0x0fULL << (i << 2))) | (value_t(t & 0x0f) << (i << 2)); }

private:
	struct lookup {
		int raw; // base row (16-bit raw)
		int left; // left operation
		int right; // right operation
		int score; // merge reward

		void init(const int& r) {
			raw = r;

			int V[4] = { (r >> 0) & 0x0f, (r >> 4) & 0x0f, (r >> 8) & 0x0f, (r >> 12) & 0x0f };
			int L[4] = { V[0], V[1], V[2], V[3] };
			int R[4] = { V[3], V[2], V[1], V[0] }; // mirrored

			score = mvleft(L);
			left = ((L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12));

			score = mvleft(R); std::reverse(R, R + 4);
			right = ((R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12));
		}

		inline void move_left(value_t& raw, int& sc, const int& i) const {
			raw |= value_t(left) << (i << 4);
			sc += score;
		}

		inline void move_right(value_t& raw, int& sc, const int& i) const {
			raw |= value_t(right) << (i << 4);
			sc += score;
		}

		static int mvleft(int row[]) {
			int top = 0;
			int tmp = 0;
			int score = 0;

			for (int i = 0; i < 4; i++) {
				int tile = row[i];
				if (tile == 0) continue;
				row[i] = 0;
				if (tmp != 0) {
					if (tile == tmp) {
						tile = tile + 1;
						row[top++] = tile;
						score += (1 << tile);
						tmp = 0;
					} else {
						row[top++] = tmp;
						tmp = tile;
					}
				} else {
					tmp = tile;
				}
			}
			if (tmp != 0) row[top] = tmp;
			return score;
		}

		struct init_t {
			init_t(lookup* c) {
				for (size_t i = 0; i < 65536; i++)
					c[i].init(i);
			}
		};

		static const lookup& find(const int& row) {
			static lookup cache[65536];
			static init_t init(cache);
			return cache[row];
		}
	};

public:
	inline int move_left() {
		value_t move = 0;
		value_t prev = raw;
		int score = 0;
		lookup::find(fetch(0)).move_left(move, score, 0);
		lookup::find(fetch(1)).move_left(move, score, 1);
		lookup::find(fetch(2)).move_left(move, score, 2);
		lookup::find(fetch(3)).move_left(move, score, 3);
		raw = move;
		return (move != prev) ? score : -1;
	}
	inline int move_right() {
		value_t move = 0;
		value_t prev = raw;
		int score = 0;
		lookup::find(fetch(0)).move_right(move, score, 0);
		lookup::find(fetch(1)).move_right(move, score, 1);
		lookup::find(fetch(2)).move_right(move, score, 2);
		lookup::find(fetch(3)).move_right(move, score, 3);
		raw = move;
		return (move != prev) ? score : -1;
	}
	inline int move_up() {
		rotate_right();
		int score = move_right();
		rotate_left();
		return score;
	}
	inline int move_down() {
		rotate_right();
		int score = move_left();
		rotate_left();
		return score;
	}
	inline int move(const int& opcode) { // 0:up 1:right 2:down 3:left
		switch (opcode) {
		case 0: return move_up();
		case 1: return move_right();
		case 2: return move_down();
		case 3: return move_left();
		default: return move((opcode % 4 + 4) % 4);
		}
	}

	inline void transpose() {
		raw = (raw & 0xf0f00f0ff0f00f0fULL) | ((raw & 0x0000f0f00000f0f0ULL) << 12) | ((raw & 0x0f0f00000f0f0000ULL) >> 12);
		raw = (raw & 0xff00ff0000ff00ffULL) | ((raw & 0x00000000ff00ff00ULL) << 24) | ((raw & 0x00ff00ff00000000ULL) >> 24);
	}
	inline void mirror() {
		raw = ((raw & 0x000f000f000f000fULL) << 12) | ((raw & 0x00f000f000f000f0ULL) << 4)
			| ((raw & 0x0f000f000f000f00ULL) >> 4) | ((raw & 0xf000f000f000f000ULL) >> 12);
	}
	inline void flip() {
		raw = ((raw & 0x000000000000ffffULL) << 48) | ((raw & 0x00000000ffff0000ULL) << 16)
			| ((raw & 0x0000ffff00000000ULL) >> 16) | ((raw & 0xffff000000000000ULL) >> 48);
	}

	inline void rotate_right() { transpose(); mirror(); } // clockwise
	inline void rotate_left() { transpose(); flip(); } // counterclockwise
	inline void reverse() { mirror(); flip(); }

	inline void rotate(const int& r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_right(); break;
		case 2: reverse(); break;
		case 3: rotate_left(); break;
		}
	}

	inline void init() { raw = 0; popup(); popup(); }
	inline void popup() { // add a new random 2-tile or 4-tile
		int space[16], num = 0;
		for (int i = 0; i < 16; i++)
			if (at(i) == 0) {
				space[num++] = i;
			}
		if (num)
			set(space[rand() % num], rand() % 10 ? 1 : 2);
	}

    friend std::ostream& operator <<(std::ostream& out, const board& b) {
		char buff[32];
		out << "+------------------------+" << std::endl;
		for (int i = 0; i < 16; i += 4) {
			snprintf(buff, sizeof(buff), "|%6u%6u%6u%6u|",
				(1 << b.at(i + 0)) & -2u,
				(1 << b.at(i + 1)) & -2u,
				(1 << b.at(i + 2)) & -2u,
				(1 << b.at(i + 3)) & -2u);
			out << buff << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}

private:
	value_t raw;
};

/**
 * feature and weight table for temporal difference learning
 */
class feature {
public:
	feature(const size_t& len) : length(len), weight(alloc(len)) {}
	virtual ~feature() { delete[] weight; }
	inline float& operator[] (const size_t& i) { return weight[i]; }
	size_t size() const { return length; }
	static std::vector<feature*>& list() {
		static std::vector<feature*> feats;
		return feats;
	}
	friend std::ostream& operator <<(std::ostream& out, const feature& w) {
		std::string name = w.name();
		int len = name.length();
		out.write(reinterpret_cast<char*>(&len), sizeof(int));
		out.write(name.c_str(), len);
		float* weight = w.weight;
		size_t size = w.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size_t));
		out.write(reinterpret_cast<char*>(weight), sizeof(float) * size);
		return out;
	}
	friend std::istream& operator >>(std::istream& in, feature& w) {
		std::string name;
		int len = 0;
		in.read(reinterpret_cast<char*>(&len), sizeof(int));
		name.resize(len);
		in.read(&name[0], len);
		if (name != w.name()) {
			error << "unexpected feature: " << name << " (" << name << " is expected)" << std::endl;
			std::exit(1);
		}
		float* weight = w.weight;
		size_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
		if (size != w.size()) std::exit(1);
		in.read(reinterpret_cast<char*>(weight), sizeof(float) * size);
		if (!in) {
			error << "unexpected end of binary" << std::endl;
			std::exit(1);
		}
		return in;
	}

public: // should be implemented
	virtual float estimate(const board& b) = 0;
	virtual float update(const board& b, const float& upd) = 0;
	virtual std::string name() const = 0;
protected:
	static float* alloc(size_t num) {
		static size_t total = 0;
		static size_t limit = 1 << 30; // 1G memory
		try {
			total += num;
			if (total > limit) throw std::bad_alloc();
			return new float[num]();
		} catch (std::bad_alloc&) {
			error << "memory limit exceeded" << std::endl;
			std::exit(-1);
		}
		return NULL;
	}
	size_t length;
	float* weight;
};

/**
 * the pattern feature
 */
template<int N>
class pattern : public feature {
public:
	pattern(int t0, ...) : feature(1 << (N * 4)) {
		va_list ap;
		va_start(ap, t0);
		patt[0] = t0;
		for (int n = 1; n < N; n++) {
			patt[n] = va_arg(ap, int);
		}
		va_end(ap);

		int isopatt[N];
		for (int i = 0; i < 8; i++) { // rotate and mirror the pattern
			board iso = 0xfedcba9876543210ull;
			if (i >= 4) iso.mirror();
			iso.rotate(i);
			for (int n = 0; n < N; n++)
				isopatt[n] = iso.at(patt[n]);
			isomorphic[i].init(isopatt);
		}

		info << name() << " initialized, size = " << length;
		if (length >= (1 << 30)) {
			info << " (" << (length >> 30) << "G)";
		} else if (length >= (1 << 20)) {
			info << " (" << (length >> 20) << "M)";
		} else if (length >= (1 << 10)) {
			info << " (" << (length >> 10) << "K)";
		}
		info << std::endl;
	}
	virtual ~pattern() {}

	virtual float estimate(const board& b) {
		debug << name() << " estimate: " << std::endl << b;
		float value = 0;
		for (int i = 0; i < 8; i++)
			value += (operator [](isomorphic[i][b]));
		return value;
	}
	virtual float update(const board& b, const float& v) {
		debug << name() << " update: " << v << std::endl;
		float value = 0;
		for (int i = 0; i < 8; i++)
			value += (operator [](isomorphic[i][b]) += v);
		return value;
	}
	virtual std::string name() const {
		std::stringstream ss;
		ss << N << "-tuple pattern " << std::hex;
		for (int i = 0; i < N; i++)
			ss << patt[i];
		return ss.str();
	}
private:
	struct indexer {
		int patt[N];
		void init(int p[N]) { std::copy(p, p + N, patt); }
		inline size_t operator[](const board& b) const {
			size_t index = 0;
			for (int i = 0; i < N; i++)
				index |= b.at(patt[i]) << (4 * i);
			return index;
		}
		std::string name() const {
			std::stringstream ss;
			ss << std::hex;
			for (int i = 0; i < N; i++)
				ss << patt[i];
			return ss.str();
		}
	};

	int patt[N];
	indexer isomorphic[8];
};

/**
 * after-state wrapper
 */
class state {
public:
	state(const int& opcode) : opcode(opcode), value(-std::numeric_limits<float>::max()), score(-1) {}
	state(const state& st) : opcode(st.opcode), before(st.before), after(st.after), value(st.value), score(st.score) {}
	state(const board& b, const int& opcode) : opcode(opcode), value(0), score(-1) { assign(b); }

	board after_state() const { return after; }
	board before_state() const { return before; }
	int   merge_score() const { return score; }
	float estimated_value() const { return value; }

	state& assign(const board& b) {
		debug << "assign " << std::endl << before;
		after = before = b;
		score = after.move(opcode);
		return *this;
	}

	state& estimate() {
		debug << "estimate " << std::endl << before;
		if (score != -1) {
			value = score;
			for (size_t i = 0; i < feature::list().size(); i++)
				value += feature::list()[i]->estimate(after);
		} else {
			value = -std::numeric_limits<float>::max();
		}
		return *this;
	}

	state& update(const float& exact, const float& alpha = 0.001) {
		debug << "update " << exact << " (" << alpha << ")" << std::endl;
		float error = exact - (value - score);
		float update = alpha * error;
		value = score;
		for (size_t i = 0; i < feature::list().size(); i++)
			value += feature::list()[i]->update(after, update);
		return *this;
	}

	bool is_valid() const {
		if (std::isnan(value)) {
			error << "numeric exception" << std::endl;
			std::exit(1);
		}
		return score != -1;
	}

	const char* name() const {
		static const char* opname[4] = { "up", "right", "down", "left" };
		return opname[opcode];
	}

    friend std::ostream& operator <<(std::ostream& out, const state& st) {
		out << "moving " << st.name() << ", reward = " << st.score;
		if (st.is_valid()) {
			info << ", value = " << st.value << std::endl << st.after;
		} else {
			info << " (invalid)" << std::endl;
		}
		return out;
	}
private:
	int opcode;
	board before;
	board after;
	float value;
	int score;
};

int main(int argc, const char* argv[]) {
	info << "TDL2048-Demo" << std::endl;

	// initialize the learning parameters
	float alpha = 0.001;
	size_t total = 100000;
	unsigned int seed;
    __asm__ __volatile__ ("rdtsc" : "=a" (seed));
	std::srand(seed);

	info << "alpha = " << alpha << std::endl;
	info << "total = " << total << std::endl;
	info << "seed = " << seed << std::endl;

	// initialize the patterns
	feature::list().push_back(new pattern<4>(0, 1, 2, 3));
	feature::list().push_back(new pattern<4>(4, 5, 6, 7));

	// initialize the weight table binary path
	std::string load = "";
	std::string save = "";

	// load weight table from binary file
	std::ifstream in;
	in.open(load.c_str(), std::ios::in | std::ios::binary);
	if (in.is_open()) {
		size_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		if (size != feature::list().size()) {
			error << "unexpected feature count: " << size
					<< " (" << feature::list().size() << " is expected)" << std::endl;
			std::exit(1);
		}
		for (size_t i = 0; i < size; i++) {
			in >> *(feature::list()[i]);
			info << feature::list()[i]->name() << " is loaded from " << load << std::endl;
		}
		in.close();
	}

	std::vector<state> path;
	path.reserve(20000);
	int scores[1000];
	int maxtile[1000];

	for (size_t n = 1; n <= total; n++) {

		// play an episode
		int score = 0;
		board b;
		b.init();
		state after[4] = { 0 /* up */, 1 /* right */, 2 /* down */, 3 /* left */ };
		while (true) {
			debug << "beforestate" << std::endl << b;

			// try to find a best move
			state* best = after;
			for (state* move = after; move != after + 4; move++) {
				move->assign(b);
				move->estimate();
				if (move->estimated_value() > best->estimated_value())
					best = move;
				debug << "try " << *move;
			}

			if (best->is_valid()) {
				debug << "best " << *best;
				path.push_back(*best);
				score += best->merge_score();
				b = best->after_state();
				b.popup();
			} else {
				debug << "gameover, ";
				break;
			}
		}

		// update the weight table by TD(0)
		float exact = 0;
		while (path.size()) {
			path.back().update(exact, alpha);
			exact = path.back().estimated_value();
			path.pop_back();
		}

		// statistics
		int ep = (n - 1) % 1000;
		scores[ep] = score;
		maxtile[ep] = 0;
		for (int i = 0; i < 16; i++)
			maxtile[ep] = std::max(maxtile[ep], b.at(i));

		// show the training process
		if (n % 1000 == 0) {
			float sum = 0;
			int max = 0;
			int stat[16] = { 0 };
			for (int i = 0; i < 1000; i++) {
				sum += scores[i];
				max = std::max(max, scores[i]);
				stat[maxtile[i]]++;
			}
			float mean = sum / 1000;
			info << n;
			info << "\t" "mean = " << mean;
			info << "\t" "max = " << max;
			info << std::endl;

			int t = 1;
			while (stat[t] == 0) t++;
			for (int c = 0; c < 1000; t++) {
				c += stat[t];
				info << "\t" << ((1 << t) & -2u) << "\t" << (stat[t] * 0.1) << "%\t(" << (c * 0.1) << "%)" << std::endl;
			}

		}
	}

	// save weight table to binary file
	std::ofstream out;
	out.open(save.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
	if (out.is_open()) {
		size_t size = feature::list().size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (size_t i = 0; i < size; i++) {
			out << *(feature::list()[i]);
			info << feature::list()[i]->name() << " is saved to " << save << std::endl;
		}
		out.flush();
		out.close();
	}

	return 0;
}
