from sys import exit
import random
import datetime
player2 = 'O' #Global Variables
player1 = 'X'
win_combos = [["1", "2", '3'], ['4', '5', '6'], ['7', '8', '9'], ['1', '4', '7'], ['2', '5', '8'],
['3', '6', '9'], ['1', '5', '9'], ['3', '5', '7']]
avail_squares = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
board_status = {1: '1', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}


def initialize(): #game setup, player picks X or O.
    print ("-" * 40 + '\n' + "-" * 40)* 10
    print "-----WELCOME TO TIC-TAC-TOE-----"
    print BOARD()
    return P1()

def BOARD(): #manipulates the image of the board, displays it to the CP
#dictionary, coordinates = key, "-, X, O"= value
    board =  '''                |'%s'|    |'%s'|    |'%s'|

                |'%s'|    |'%s'|    |'%s'|

                |'%s'|    |'%s'|    |'%s'|''' % (board_status[1], board_status[2],
    board_status[3], board_status[4], board_status[5],
    board_status[6], board_status[7], board_status[8], board_status[9])

    return board

def BOARD_SCAN(win_combos, turn):
#determines if a defensive or offensive move must be made
    global player2
    global avail_squares
    global player1
    for combo in win_combos:
        if combo.count(player1) == 3:
            return VICTOR(player1)
        elif combo.count(player2) == 3:
            return VICTOR(player2)
    if len(avail_squares) == 0:
        return tie()

    if turn == player1:
        return P1()
    elif turn == player2:
        return P2()


def VICTOR(winner):
    print ("-" * 40 + '\n' + "-" * 40) * 3

    #print ("*" * 10) * 3
    if winner == player1:
        print "X WINS!!!!"
        print "THANKS FOR PLAYING. GOODBYE"
        exit()

    if winner == player2:
        print "O WINS!!!!"
        print "THANKS FOR PLAYING. GOODBYE"
        exit()


def P2(): #players turn, appends the newly occupied square to list '0 squares'.
 #Deletes the square from list 'avail_squares'

    print " O's TURN"
    print "Select a NUMBER on the board to make your move."
    number = raw_input('>')
    if len(number) < 1:
        print 'PLEASE ENTER A NUMBER'
        return P2()

    global win_combos
    global board_status
    global avail_squares
    global player
    if board_status[int(number)].isdigit() == False:
        print "Sorry, You can't go there. Try again!"
        return P2()
    else:
        for sequence in win_combos:
            for value in sequence:
                if value == number:
                    val_index = sequence.index(number)
                    sequence[val_index] = player2

            for item in avail_squares:
                if item == number:
                    space_index = avail_squares.index(item)
                    del(avail_squares[space_index])
            for pair in board_status:
                board_status[int(number)] = player2

    print BOARD()
    return BOARD_SCAN(win_combos, player1)


def P1(): #players turn, appends the newly occupied square to list '0 squares'.
 #Deletes the square from list 'avail_squares'

    print " X's TURN"
    print "Select a NUMBER on the board to make your move."
    number = raw_input('>')
    if len(number) <1:
        print 'PLEASE ENTER A NUMBER'
        return P1()
    global win_combos
    global board_status
    global avail_squares
    global player
    if board_status[int(number)].isdigit() == False:
        print "Sorry, You can't go there. Try again!"
        return P1()
    else:
        for sequence in win_combos:
            for value in sequence:
                if value == number:
                    val_index = sequence.index(number)
                    sequence[val_index] = player1

            for item in avail_squares:
                if item == number:
                    space_index = avail_squares.index(item)
                    del(avail_squares[space_index])
            for pair in board_status:
                board_status[int(number)] = player1

    print BOARD()
    return BOARD_SCAN(win_combos, player2)

def tie():
    print ("-" * 40 + '\n' + "-" * 40)* 10
    print "It's a tie!"
    print "THANKS FOR PLAYING. GOODBYE"
    exit()

initialize()
