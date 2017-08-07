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
 * [3] Oka, Kazuto, and Kiminori Matsuzaki. "Systematic selection of n-tuple networks for 2048."
 * International Conference on Computers and Games. Springer International Publishing, 2016.
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
	output& operator <<(const type& v) {
		if (enable) out << v;
		return *this;
	}
	output& operator <<(std::ostream& (*pf)(std::ostream&)) {
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

	board(const value_t& raw = 0) : raw(raw) {}
	board(const board& b) : raw(b.raw) {}
	operator value_t&() { return raw; }
	bool operator ==(const board& b) const { return raw == b.raw; }
	bool operator !=(const board& b) const { return raw != b.raw; }


	int  fetch(const int& i) const { return ((raw >> (i << 4)) & 0xffff); }
	void place(const int& i, const int& r) { raw = (raw & ~(0xffffULL << (i << 4))) | (value_t(r & 0xffff) << (i << 4)); }
	int  at(const int& i) const { return (raw >> (i << 2)) & 0x0f; }
	void set(const int& i, const int& t) { raw = (raw & ~(0x0fULL << (i << 2))) | (value_t(t & 0x0f) << (i << 2)); }

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

		void move_left(value_t& raw, int& sc, const int& i) const {
			raw |= value_t(left) << (i << 4);
			sc += score;
		}

		void move_right(value_t& raw, int& sc, const int& i) const {
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

		lookup() {
			static int row = 0;
			init(row++);
		}

		static const lookup& find(const int& row) {
			static const lookup cache[65536];
			return cache[row];
		}
	};

public:
	int move_left() {
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
	int move_right() {
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
	int move_up() {
		rotate_right();
		int score = move_right();
		rotate_left();
		return score;
	}
	int move_down() {
		rotate_right();
		int score = move_left();
		rotate_left();
		return score;
	}
	int move(const int& opcode) {
		switch (opcode) {
		case 0: return move_up();
		case 1: return move_right();
		case 2: return move_down();
		case 3: return move_left();
		default: return -1;
		}
	}

	void transpose() {
		raw = (raw & 0xf0f00f0ff0f00f0fULL) | ((raw & 0x0000f0f00000f0f0ULL) << 12) | ((raw & 0x0f0f00000f0f0000ULL) >> 12);
		raw = (raw & 0xff00ff0000ff00ffULL) | ((raw & 0x00000000ff00ff00ULL) << 24) | ((raw & 0x00ff00ff00000000ULL) >> 24);
	}
	void mirror() {
		raw = ((raw & 0x000f000f000f000fULL) << 12) | ((raw & 0x00f000f000f000f0ULL) << 4)
			| ((raw & 0x0f000f000f000f00ULL) >> 4) | ((raw & 0xf000f000f000f000ULL) >> 12);
	}
	void flip() {
		raw = ((raw & 0x000000000000ffffULL) << 48) | ((raw & 0x00000000ffff0000ULL) << 16)
			| ((raw & 0x0000ffff00000000ULL) >> 16) | ((raw & 0xffff000000000000ULL) >> 48);
	}

	void rotate_right() { transpose(); mirror(); } // clockwise
	void rotate_left() { transpose(); flip(); } // counterclockwise
	void reverse() { mirror(); flip(); }

	void rotate(const int& r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_right(); break;
		case 2: reverse(); break;
		case 3: rotate_left(); break;
		}
	}

	void init() { raw = 0; popup(); popup(); }
	void popup() { // add a new random 2-tile or 4-tile
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
	float& operator[] (const size_t& i) { return weight[i]; }
	size_t size() const { return length; }

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
 * including isomorphic (rotate/mirror)
 *
 * index:
 *  0  1  2  3
 *  4  5  6  7
 *  8  9 10 11
 * 12 13 14 15
 *
 * usage:
 *  pattern<4>(0, 1, 2, 3)
 *  pattern<6>(0, 1, 2, 3, 4, 5)
 */
template<int N>
class pattern : public feature {
public:
	pattern(int t0, ...) : feature(1 << (N * 4)), iso_last(8) {
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

		info << name() << ", size = " << length;
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
		for (int i = 0; i < iso_last; i++)
			value += (operator [](isomorphic[i][b]));
		return value;
	}
	virtual float update(const board& b, const float& v) {
		debug << name() << " update: " << v << std::endl;
		float value = 0;
		for (int i = 0; i < iso_last; i++)
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

	/*
	 * set the isomorphic of this pattern
	 * 0: disable isomorphic
	 * 4: enable rotation
	 * 8: enable rotation and reflection
	 */
	void set_isomorphic(const int& i = 8) { iso_last = i; }
private:
	struct indexer {
		int patt[N];
		void init(int p[N]) { std::copy(p, p + N, patt); }
		size_t operator[](const board& b) const {
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
	int iso_last;
};

/**
 * before state and after state wrapper
 */
class state {
public:
	state(const int& opcode = -1)
		: opcode(opcode), score(-1), esti(0) {}
	state(const board& b, const int& opcode = -1)
		: opcode(opcode), score(-1), esti(0) { assign(b); }
	state(const state& st)
		: before(st.before), after(st.after), opcode(st.opcode), score(st.score), esti(st.esti) {}

	board after_state() const { return after; }
	board before_state() const { return before; }
	float value() const { return esti; }
	int reward() const { return score; }
	int action() const { return opcode; }
	const char* name() const {
		static const char* opname[4] = { "up", "right", "down", "left" };
		return (opcode >= 0 && opcode < 4) ? opname[opcode] : "none";
	}

	bool assign(const board& b) {
		debug << "assign " << std::endl << b;
		after = before = b;
		score = after.move(opcode);
		return score != -1;
	}

	void set_before_state(const board& b) { before = b; }
	void set_after_state(const board& b) { after = b; }
	void set_value(const float& v) { esti = v; }
	void set_reward(const int& r) { score = r; }
	void set_action(const int& a) { opcode = a; }

	bool is_valid() const {
		if (std::isnan(esti)) {
			error << "numeric exception" << std::endl;
			std::exit(1);
		}
		return after != before && opcode != -1 && score != -1;
	}

    friend std::ostream& operator <<(std::ostream& out, const state& st) {
		out << "moving " << st.name() << ", reward = " << st.score;
		if (st.is_valid()) {
			info << ", value = " << st.esti << std::endl << st.after;
		} else {
			info << " (invalid)" << std::endl;
		}
		return out;
	}
private:
	board before;
	board after;
	int opcode;
	int score;
	float esti;
};

class learning {
public:
	learning() {}
	~learning() { for (size_t i = 0; i < feats.size(); i++) delete feats[i]; }
	void add_feature(feature* feat) { feats.push_back(feat); }

	/**
	 * accumulate the total value of given state
	 */
	float estimate(const board& b) {
		debug << "estimate " << std::endl << b;
		float value = 0;
		for (size_t i = 0; i < feats.size(); i++)
			value += feats[i]->estimate(b);
		return value;
	}

	/**
	 * update the value of given state and return its new value
	 */
	float update(const board& b, const float& update) {
		debug << "update " << " (" << update << ")" << std::endl << b;
		float value = 0;
		for (size_t i = 0; i < feats.size(); i++)
			value += feats[i]->update(b, update);
		return 0;
	}

	/**
	 * select a best move of a before state b
	 *
	 * return should be a state whose
	 *  before_state() is b
	 *  after_state() is b's best successor (after state)
	 *  action() is the best action
	 *  reward() is the reward of performing action()
	 *  value() is the estimated value of after_state()
	 *
	 * you may simply return state() if no valid move
	 */
	state select_best_move(const board& b) {
		state after[4] = { 0, 1, 2, 3 }; // up, right, down, left
		state* best = after;
		for (state* move = after; move != after + 4; move++) {
			if (move->assign(b)) {
				move->set_value(move->reward() + estimate(move->after_state()));
				if (move->value() > best->value())
					best = move;
			} else {
				move->set_value(-std::numeric_limits<float>::max());
			}
			debug << "try " << *move;
		}
		return *best;
	}

	/**
	 * update the tuple network by an episode
	 *
	 * path is the sequence of states in this episode,
	 * the last entry in path (path.back()) is the final state
	 *
	 * for example, a 2048 games consists of
	 *  (initial) s0 --(a0,r0)--> s0' --(popup)--> s1 --(a1,r1)--> s1' --(popup)--> s2 (game over)
	 *  where sx is before state, sx' is after state
	 * its path would be
	 *  { (s0,s0',a0,r0), (s1,s1',a1,r1), (s2,-,-,-) }
	 *  see the constructor of state for more details
	 */
	void update_episode(std::vector<state>& path, const float& alpha = 0.001) {
		float exact = 0;
		for (path.pop_back(); path.size(); path.pop_back()) {
			state& move = path.back();
			float error = exact - (move.value() - move.reward());
			exact = move.reward() + update(move.after_state(), alpha * error);
		}
	}

	/**
	 * update the statistic, and display the status once in 1000 episodes
	 */
	void update_statistics(size_t n, const board& b, const int& score) {
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

	void load(const std::string& path) {
		std::ifstream in;
		in.open(path.c_str(), std::ios::in | std::ios::binary);
		if (in.is_open()) {
			size_t size;
			in.read(reinterpret_cast<char*>(&size), sizeof(size));
			if (size != feats.size()) {
				error << "unexpected feature count: " << size << " (" << feats.size() << " is expected)" << std::endl;
				std::exit(1);
			}
			for (size_t i = 0; i < size; i++) {
				in >> *(feats[i]);
				info << feats[i]->name() << " is loaded from " << path << std::endl;
			}
			in.close();
		}
	}

	void save(const std::string& path) {
		std::ofstream out;
		out.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
		if (out.is_open()) {
			size_t size = feats.size();
			out.write(reinterpret_cast<char*>(&size), sizeof(size));
			for (size_t i = 0; i < size; i++) {
				out << *(feats[i]);
				info << feats[i]->name() << " is saved to " << path << std::endl;
			}
			out.flush();
			out.close();
		}
	}

private:
	std::vector<feature*> feats;
	int scores[1000];
	int maxtile[1000];
};

int main(int argc, const char* argv[]) {
	info << "TDL2048-Demo" << std::endl;
	learning tdl;

	// initialize the learning parameters
	float alpha = 0.001;
	size_t total = 100000;
	unsigned seed; __asm__ __volatile__ ("rdtsc" : "=a" (seed));
	info << "alpha = " << alpha << std::endl;
	info << "total = " << total << std::endl;
	info << "seed = " << seed << std::endl;
	std::srand(seed);

	// initialize the patterns
	tdl.add_feature(new pattern<4>(0, 1, 2, 3));
	tdl.add_feature(new pattern<4>(4, 5, 6, 7));
	tdl.add_feature(new pattern<4>(0, 1, 4, 5));
	tdl.add_feature(new pattern<4>(1, 2, 5, 6));
	tdl.add_feature(new pattern<4>(5, 6, 9,10));

	// restore the model from the file if necessary
	tdl.load("");

	// train the model
	std::vector<state> path;
	path.reserve(20000);
	for (size_t n = 1; n <= total; n++) {
		int score = 0;

		// play an episode
		board b;
		b.init();
		while (true) {
			debug << "state" << std::endl << b;
			state best = tdl.select_best_move(b);

			if (best.is_valid()) {
				debug << "best " << best;
				score += best.reward();
				path.push_back(best);
				b = best.after_state();
				b.popup();
			} else {
				debug << "gameover, ";
				path.push_back(state(b));
				break;
			}
		}

		tdl.update_episode(path, alpha);
		tdl.update_statistics(n, b, score);
		path.clear();
	}

	// store the model into the file
	tdl.save("");

	return 0;
}
