import sys

import hlt

_productions = []
_width = -1
_height = -1


def serialize_move_set(moves):
    return_string = ""
    for move in moves:
        return_string += str(move.loc.x) + " " + str(move.loc.y) + " " + str(move.direction) + " "
    return return_string


def deserialize_map_size(input_string):
    global _width, _height

    split_string = input_string.split(" ")
    _width = int(split_string.pop(0))
    _height = int(split_string.pop(0))


def deserialize_productions(input_string):
    split_string = input_string.split(" ")

    for a in range(0, _height):
        row = []
        for b in range(0, _width):
            row.append(int(split_string.pop(0)))
        _productions.append(row)


def deserialize_map(input_string):
    split_string = input_string.split(" ")

    m = hlt.GameMap(_width, _height)

    y = 0
    x = 0
    while y != m.height:
        counter = int(split_string.pop(0))
        owner = int(split_string.pop(0))
        for a in range(0, counter):
            m.contents[y][x].owner = owner
            x += 1
            if x == m.width:
                x = 0
                y += 1

    for a in range(0, _height):
        for b in range(0, _width):
            m.contents[a][b].strength = int(split_string.pop(0))
            m.contents[a][b].production = _productions[a][b]

    return m


def send_string(to_be_sent):
    to_be_sent += '\n'

    sys.stdout.write(to_be_sent)
    sys.stdout.flush()


def get_string():
    return sys.stdin.readline().rstrip('\n')


def get_init():
    player_tag = int(get_string())
    deserialize_map_size(get_string())
    deserialize_productions(get_string())
    m = deserialize_map(get_string())

    return player_tag, m


def send_init(name):
    send_string(name)


def get_frame():
    return deserialize_map(get_string())


def send_frame(moves):
    send_string(serialize_move_set(moves))
