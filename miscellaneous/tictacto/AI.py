from sys import exit
import random

AI = '-' #Global Variables for reference
player = '-'
win_combos = [["1", "2", '3'], ['4', '5', '6'], ['7', '8', '9'], ['1', '4', '7'],
            ['2', '5', '8'], ['3', '6', '9'], ['1', '5', '9'], ['3', '5', '7']]
avail_squares = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
board_status = {1: '1', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}
last_move = []

def initialize(): #game setup, player picks X or O.
    global player
    global AI
    global win_combos

    print ("-" * 40 + '\n' + "-" * 40)* 10
    print "-----WELCOME TO TIC-TAC-TOE-----"
    print "Do you want to play as 'X' or 'O'?"
    choice = raw_input('>').upper()

    if choice == "X":
        player = "X"
        AI = "O"
        last_move.append(player)
        print BOARD()
        return P2()

    elif choice == "O":
        player = "O"
        AI = "X"
        return BOARD_SCAN(win_combos)
    else:
        print "Please type 'X' or 'O'"
        return initialize()

def BOARD():

    board =  '''                |'%s'|    |'%s'|    |'%s'|

                |'%s'|    |'%s'|    |'%s'|

                |'%s'|    |'%s'|    |'%s'|''' % (board_status[1], board_status[2],
                                                board_status[3], board_status[4],
                                                board_status[5], board_status[6],
                                                board_status[7], board_status[8],
                                                board_status[9])

    return board

def BOARD_SCAN(win_combos):
#determines if win condition is met, or if offensive/defensive move must be made
    global AI
    global avail_squares
    global player

    for combo in win_combos:
        if combo.count(player) == 3:
            return VICTOR(player)
        elif combo.count(player) == 3:
            return VICTOR(AI)
        elif len(avail_squares) == 0:
            return tie()

    for combo in win_combos:
        if combo.count(AI) == 2 and combo.count(player) == 0:
            return OFFENSE(combo)

    for combo in win_combos:
        if combo.count(player) == 2 and combo.count(AI) == 0:
            return DEFENSE(combo)
    combo = None
    return OFFENSE(combo)

def DEFENSE(combo):

    global AI
    global win_combos
    global avail_squares
    global board_status
    global last_move
    last_move.append(AI)
    for element in combo:
        if element.isdigit() == True:
            move = element
            competitor = AI

            return LIST_MANAGEMENT(move, competitor)
    return P2()

def VICTOR(winner):
    print ("-" * 40 + '\n' + "-" * 40) * 3

    if winner == AI:
        print "YOU LOSE"
        print "THANKS FOR PLAYING. GOODBYE"
        exit()

    if winner == player:
        print "YOU WIN"
        print "THANKS FOR PLAYING. GOODBYE"
        exit()

def OFFENSE(combo):
    global player
    global avail_squares
    global win_combos
    global board_status
    global last_move
    global AI

    corn_squares = ['1', '9', '7', '3']
    side_squares = ['2', '4', '6', '8']

    last_move.append(AI)
    competitor = AI
    if combo == None:
        pass
    else:
        for value in combo:
            if value.isdigit() == True:
                move = value
                competitor = AI
                return LIST_MANAGEMENT(move, competitor)

    for square in corn_squares:
        if square in avail_squares:
            move = square
            return LIST_MANAGEMENT(move, competitor)
    avail_side_square = []
    for square in side_squares:
        if square in avail_squares:
            avail_side_square.append(square)
            move = random.choice(avail_side_square)
            del avail_side_square[:]
            return LIST_MANAGEMENT(move, competitor)

    move = random.choice(avail_squares)
    return LIST_MANAGEMENT(move, competitor)

def LIST_MANAGEMENT(move, competitor):

#turns a particular number in each win_combo into an 'X' or 'O'
    for sequence in win_combos:
        for value in sequence:
            if value == move:
                val_index = sequence.index(value)
                sequence[val_index] = competitor

#deletes the square from the list of available squares
    if move in avail_squares:
        move_index = avail_squares.index(move)
        del(avail_squares[move_index])


#updates the tictacto board which gets printed to the user
    for pair in board_status:
        board_status[int(move)] = competitor

#analyzes for terminal conditions
    for combo in win_combos:
        if combo.count(player) == 3:

            return VICTOR(player)
        elif combo.count(AI) == 3:

            return VICTOR(AI)
        elif len(avail_squares) == 0: #put the tie condition at the end
            return tie()


    if last_move[-1] == AI:
        print BOARD()
        return P2()
    elif last_move[-1] == player:
        return BOARD_SCAN(win_combos)

def P2(): #players turn, appends the newly occupied square to list '0 squares'.
 #Deletes the square from list 'avail_squares'
    global win_combos
    global board_status
    global avail_squares
    global player
    global last_move

    last_move[-1] = player
    print "YOUR MOVE"
    print "Select a NUMBER on the board to make your move."

    move = raw_input('>')

    if move.isdigit() == False:
        print 'PLEASE ENTER A VAILD NUMBER'
        return P2()
    if int(move) not in range(1, 10):
        print 'PLEASE ENTER A VALID NUMBER'
        return P2()

    if board_status[int(move)].isdigit() == False:
        print "Sorry, You can't go there. Try again!"
        return P2()
    competitor = player

    return LIST_MANAGEMENT(move, competitor)

def tie():
    print ("-" * 40 + '\n' + "-" * 40)* 10
    print "It's a tie!"
    print "THANKS FOR PLAYING. GOODBYE"
    exit()

initialize()
