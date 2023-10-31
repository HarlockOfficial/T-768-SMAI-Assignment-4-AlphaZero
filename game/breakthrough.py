import random
import types


class Breakthrough:
    White = 0
    Black = 1
    Opponent = [Black, White]

    class Board:
        NoPce = -1

        def __init__(self, cols, rows):
            assert (cols >= 2)
            assert (rows >= 4)
            self._cols = cols
            self._rows = rows
            self._size = cols * rows
            self._board = [self.NoPce for _ in range(self._rows*self._cols)]
            self._hash_key = 0
            self._hash_board = [[random.getrandbits(64) for _ in range(self._rows*self._cols)] for _ in range(2)]
            self.d_nw = cols - 1
            self.d_n = cols
            self.d_ne = cols + 1
            self.d_e = 1
            self.d_se = -cols + 1
            self.d_s = -cols
            self.d_sw = -cols - 1
            self.d_w = -1
            return

        def size(self):
            return self._size

        def cols(self):
            return self._cols

        def rows(self):
            return self._rows

        def sqr(self, c, r):
            # assert(0 <= c < self.cols())
            # assert(0 <= r < self.rows())
            return r * self._cols + c

        def col(self, sqr):
            # assert(0 <= sqr < self.rows() * self.cols())
            return sqr % self._cols

        def row(self, sqr):
            # assert(0 <= sqr < self.rows() * self.cols())
            return sqr // self._cols

        def col_row(self, sqr):
            return sqr % self._cols, sqr // self._cols

        def clear(self):
            self._board = [self.NoPce for _ in range(self._rows*self._cols)]
            self._hash_key = 0
            return

        def set(self, sqr, pce):
            # assert(0 <= sqr < self.rows() * self.cols())
            if self._board[sqr] != self.NoPce:
                self._hash_key ^= self._hash_board[self._board[sqr]][sqr]
            if pce != self.NoPce:
                self._hash_key ^= self._hash_board[pce][sqr]
            self._board[sqr] = pce
            return

        def set2d(self, c, r, pce):
            # assert (0 <= c < self.cols())
            # assert (0 <= r < self.rows())
            self.set(self.sqr(c, r), pce)

        def get(self, sqr):
            # assert(0 <= sqr < self.rows() * self.cols())
            return self._board[sqr]

        def get2d(self, c, r):
            # assert(0 <= c < self.cols())
            # assert(0 <= r < self.rows())
            return self._board[r * self._cols + c]

        def get_key(self):
            return self._hash_key

        def get_pce_locations(self, p):
            pce_loc = []
            for sqr in range(self._size):
                if self._board[sqr] == p:
                    pce_loc.append(sqr)
            return pce_loc

        @staticmethod
        def col_to_str(col):
            assert col < 26
            return "abcdefghijklmnopqrstuvwxyz"[col]

        @staticmethod
        def row_to_str(row):
            assert row < 9
            return str(row + 1)

        def square_to_str(self, sq):
            return self.col_to_str(self.col(sq)) + self.row_to_str(self.row(sq))

        def move_to_str(self, move):
            return self.square_to_str(move[0]) + ('-' if move[2] == self.NoPce else 'x') + self.square_to_str(move[1])

    def __init__(self, cols, rows):
        self._board = self.Board(cols, rows)
        self._to_move = self.White
        self._pce_count = [0, 0]
        self._hash_side = random.getrandbits(64)
        self._num_moves = 0
        self.setup()
        return

    def __repr__(self):
        s = ""
        for r in range(self._board.rows()-1, -1, -1):
            s += self._board.row_to_str(r) + '|'
            for c in range(self._board.cols()):
                pce = self._board.get2d(c, r)
                if pce == self.White:
                    s += 'w'
                elif pce == self.Black:
                    s += 'b'
                else:
                    s += '.'
            s += '|\n'
        s += '  '
        for c in range(self._board.cols()):
            s += self._board.col_to_str(c)
        s += '\n'
        if self._to_move == self.White:
            s += 'W'
        else:
            s += 'B'
        s += ' ' + str(self._pce_count[0]) + ' ' + str(self._pce_count[1])
        return s

    def get_to_move(self):
        return self._to_move

    def get_pce_count(self):
        return self._pce_count

    def get_board(self):
        return self._board

    def get_move_no(self):
        return self._num_moves

    def setup(self):
        self._to_move = self.White
        self._board.clear()
        for c in range(self._board.cols()):
            self._board.set2d(c, 0, self.White)
            self._board.set2d(c, 1, self.White)
            self._board.set2d(c, self._board.rows()-2, self.Black)
            self._board.set2d(c, self._board.rows()-1, self.Black)
        self._pce_count[0] = self._pce_count[1] = 2 * self._board.cols()
        self._num_moves = 0
        return

    def setup_fen(self, fen):
        assert len(fen) == self._board.size() + 2
        self._board.clear()
        self._pce_count[0] = self._pce_count[1] = 0
        i = 0
        for r in range(self._board.rows() - 1, -1, -1):
            for c in range(self._board.cols()):
                if fen[i] == 'w':
                    self._board.set2d(c, r, self.White)
                    self._pce_count[0] += 1
                elif fen[i] == 'b':
                    self._board.set2d(c, r, self.Black)
                    self._pce_count[1] += 1
                else:
                    assert fen[i] == '.'
                i += 1
        assert fen[i] == ' '
        i += 1
        if fen[i] == 'w':
            self._to_move = self.White
        elif fen[i] == 'b':
            self._to_move = self.Black
        else:
            assert False
        self._num_moves = 0
        return

    def to_fen(self):
        chars = []
        for r in range(self._board.rows() - 1, -1, -1):
            for c in range(0, self._board.cols()):
                p = self._board.get2d(c, r)
                if p == self.White:
                    ch = 'w'
                elif p == self.Black:
                    ch = 'b'
                else:
                    ch = '.'
                chars.append(ch)
        chars.append(' ')
        chars.append('w' if self._to_move == self.White else 'b')
        return ''.join(chars)

    def is_terminal(self):
        if self._pce_count[0] == 0 or self._pce_count[1] == 0:
            return True
        for c in range(self._board.cols()):
            if self._board.get2d(c, 0) == self.Black or self._board.get2d(c, self._board.rows()-1) == self.White:
                return True
        return False

    def generate(self, shuffle=False):
        moves = []
        board = self._board
        if self._to_move == self.White:
            for s in reversed(board.get_pce_locations(self.White)):
                c, r = board.col_row(s)
                if c > 0:
                    s_to = s + board.d_nw
                    if board.get(s_to) != self.White:
                        moves.append((s, s_to, self._board.get(s_to)))
                s_to = s + board.d_n
                if board.get(s_to) == self._board.NoPce:
                    moves.append((s, s_to, board.NoPce))
                if c < board.cols() - 1:
                    s_to = s + board.d_ne
                    if board.get(s_to) != self.White:
                        moves.append((s, s_to, board.get(s_to)))
        else:
            for s in board.get_pce_locations(self.Black):
                c, r = board.col_row(s)
                if c > 0:
                    s_to = s + board.d_sw
                    if board.get(s_to) != self.Black:
                        moves.append((s, s_to, self._board.get(s_to)))
                s_to = s + board.d_s
                if board.get(s_to) == self._board.NoPce:
                    moves.append((s, s_to, board.NoPce))
                if c < board.cols() - 1:
                    s_to = s + board.d_se
                    if board.get(s_to) != self.Black:
                        moves.append((s, s_to, board.get(s_to)))
        if shuffle:
            random.shuffle(moves)
        return moves

    def make(self, move):
        sqr_from, sqr_to, _ = move
        if self._board.get(sqr_to) != self._board.NoPce:
            self._pce_count[self.Opponent[self._to_move]] -= 1
        self._board.set(sqr_to, self._board.get(sqr_from))
        self._board.set(sqr_from, self._board.NoPce)
        self._to_move = self.Opponent[self._to_move]
        self._num_moves += 1
        return

    def retract(self, move):
        sqr_from, sqr_to, pce = move
        self._num_moves -= 1
        self._to_move = self.Opponent[self._to_move]
        self._board.set(sqr_from, self._board.get(sqr_to))
        self._board.set(sqr_to, pce)
        if pce != self._board.NoPce:
            self._pce_count[self.Opponent[self._to_move]] += 1
        return

    def get_key(self):
        if self._to_move == self.White:
            return self._board.get_key()
        else:
            return self._board.get_key() ^ self._hash_side
