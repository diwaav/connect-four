import numpy as np

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    # copy from ConnectFour.py: updates board with given coin
    def update_board(self, board, move, player_num):
        if 0 in board[:,move]:
            update_row = -1
            for row in range(1, board.shape[0]):
                update_row = -1
                if board[row, move] > 0 and board[row-1, move] == 0:
                    update_row = row-1
                elif row==(board.shape[0]-1) and board[row, move] == 0:
                    update_row = row

                if update_row >= 0:
                    board[update_row, move] = player_num
                    return board

    # copy from ConnectFour.py: check if game is completed or not
    def game_completed(self, board, player_num):
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True
            return False
        return check_horizontal(board) or check_verticle(board) or check_diagonal(board)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        # infinities
        posinf = float('inf')
        neginf = float('-inf')

        # finds max value for a player
        def max_value(bo, d, a, b, maximizingplayer):
            v = neginf
            for col in range(bo.shape[1]):
                if 0 in bo[:, col]:
                    temp = bo.copy()
                    player = 1
                    if self.player_number == 1:
                        player = 2
                    temp = self.update_board(temp, col, player)
                    v = max(v, alphabeta(temp, d - 1, a, b, not maximizingplayer))
                if v >= b:
                    return v
                a = max(a, v)
            return v

        # finds min value for a player
        def min_value(bo, d, a, b, maximizingplayer):
            v = posinf
            for col in range(bo.shape[1]):
                if 0 in bo[:, col]:
                    temp = bo.copy()
                    temp = self.update_board(temp, col, self.player_number)
                    v = min(v, alphabeta(temp, d - 1, a, b, maximizingplayer))
                if v <= a:
                    return v
                b = min(b, v)
            return v

        # main alphabeta function that finds heuristic at some depth for a board
        def alphabeta(bo, d, a, b, maximizingplayer):
            if (d == 0) or self.game_completed(bo, self.player_number):
                return self.evaluation_function(bo)
            if maximizingplayer:
                return max_value(bo, d, a, b, maximizingplayer)
            else:
                return min_value(bo, d, a, b, maximizingplayer)

        # try different moves and choose column with best utility/heuristic
        util = 0
        col = 0
        depth = 4
        for c in range(7):
            if 0 in board[:,c]:
                temp = board.copy()
                temp = self.update_board(temp, c, self.player_number)
                heu = alphabeta(temp, depth, posinf, neginf, True)
                if heu > util:
                    col = c
                    util = heu
        return col

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        neginf = float('-inf')

        # finds heuristic value for a board
        def value(bo, d, m, player, op):
            if (d == 0) or (self.game_completed(bo, player)):
                return self.evaluation_function(bo)
            if m:
                return max_value(bo, d, m, player, op)
            if not m:
                return exp_value(bo, d, m, player, op)

        # max value for a board
        def max_value(bo, d, m, player, op):
            v = neginf
            for column in range(bo.shape[1]):
                if 0 in bo[:, column]:
                    tempboard = bo.copy()
                    tempboard = self.update_board(tempboard, column, op)
                    v = max(v, value(tempboard, d - 1, not m, player, op))
            return v

        # finds expected value of a move given board... assume equal probability for successors
        def exp_value(bo, d, m, player, op):
            v = 0
            valid_cols = []
            p = 0
            for c in range(bo.shape[1]):
                if 0 in bo[:, c]:
                    valid_cols.append(c)
            for col in valid_cols:
                t = bo.copy()
                t = self.update_board(t, col, player)
                p = 1.0 / len(valid_cols)
                v += p * value(t, d - 1, m, player, op)
            return v

        # try different moves and choose whichever gives best utility/heuristic
        util = 0
        col = 0
        depth = 1
        opp_player = 1
        if self.player_number == 1:
            opp_player = 2
        for c in range(7):
            if 0 in board[:, c]:
                temp = board.copy()
                temp = self.update_board(temp, c, self.player_number)
                heu = value(temp, depth, True, self.player_number, opp_player)
                if heu > util:
                    col = c
                    util = heu
        return col

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        # go through each row and add to the heuristic
        # value of a player for the given board. 3 in a row are the closest to the
        # possibility of having 4 in a row for win, so they are weighted heavier than
        # two in a row.
        def check_horizontal(b, player):
            three = 0
            hPlayer = 0
            for r in b:
                # now im looking at row r
                if ('{0} {0} {0} 0'.format(player) in str(r))\
                        or ('0 {0} {0} {0}'.format(player) in str(r))\
                        or ('{0} 0 {0} {0}'.format(player) in str(r))\
                        or ('{0} {0} 0 {0}'.format(player) in str(r)):
                    hPlayer = hPlayer * 4
                    three += 1
                if ('{0} {0} 0 0'.format(player) in str(r))\
                        or ('0 {0} {0} 0'.format(player) in str(r))\
                        or ('0 0 {0} {0}'.format(player) in str(r))\
                        or ('0 {0} 0 {0}'.format(player) in str(r))\
                        or ('{0} 0 {0} 0'.format(player) in str(r))\
                        or ('{0} 0 0 {0}'.format(player) in str(r)):
                    hPlayer += 1
            return hPlayer + three

        # same as horizontal heuristic check, but checking verticals for columns
        def check_vertical(b, player):
            return check_horizontal(b.T, player)

        # same as horizontal check but checking diagonals
        def check_diagonal(b, player):
            three = 0
            hPlayer = 0
            pb = []
            # l->r diagonal
            for i in range(-2, 7):
                pb.append(b.diagonal(i).tolist())
            # r->l diagonal
            b = np.fliplr(b)
            for i in range(-2, 7):
                pb.append(b.diagonal(i).tolist())
            h = 0
            for r in b:
                # now im looking at row r
                if ('{0} {0} {0} 0'.format(player) in str(r)) \
                        or ('0 {0} {0} {0}'.format(player) in str(r)) \
                        or ('{0} 0 {0} {0}'.format(player) in str(r)) \
                        or ('{0} {0} 0 {0}'.format(player) in str(r)):
                    hPlayer = hPlayer * 4
                    three += 1
                if ('{0} {0} 0 0'.format(player) in str(r)) \
                        or ('0 {0} {0} 0'.format(player) in str(r)) \
                        or ('0 0 {0} {0}'.format(player) in str(r)) \
                        or ('0 {0} 0 {0}'.format(player) in str(r)) \
                        or ('{0} 0 {0} 0'.format(player) in str(r)) \
                        or ('{0} 0 0 {0}'.format(player) in str(r)):
                    hPlayer += 1
            return h + three

        if self.player_number == 1:
            # if player 1, weigh player 1's heuristics more than opposite player's heuristics
            ph = check_horizontal(board, 1)
            oh = check_horizontal(board, 2)
            pv = check_vertical(board, 1)
            ov = check_vertical(board, 2)
            pd = check_diagonal(board, 1)
            od = check_diagonal(board, 2)
            return (2*ph - (1/3)*oh) + (2*pv - (1/3)*ov) + (2*pd - (1/3)*od)
        else:
            # if player 2, weigh player 2's heuristics more than opposite player's heuristics
            ph = check_horizontal(board, 2)
            oh = check_horizontal(board, 1)
            pv = check_vertical(board, 2)
            ov = check_vertical(board, 1)
            pd = check_diagonal(board, 2)
            od = check_diagonal(board, 1)
            return (2*ph - (1/3)*oh) + (2*pv - (1/3)*ov) + (2*pd - (1/3)*od)


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

