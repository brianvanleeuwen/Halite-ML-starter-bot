import copy

STILL = 0
NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4

DIRECTIONS = [a for a in range(0, 5)]
CARDINALS = [a for a in range(1, 5)]

ATTACK = 0
STOP_ATTACK = 1


class Location:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Site:
    def __init__(self, owner=0, strength=0, production=0):
        self.owner = owner
        self.strength = strength
        self.production = production


class Move:
    def __init__(self, loc=0, direction=0):
        self.loc = loc
        self.direction = direction


class GameMap:
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.contents = []

        for y in range(0, self.height):
            row = []
            for x in range(0, self.width):
                row.append(Site(0, 0, 0))
            self.contents.append(row)

    def in_bounds(self, l):
        return l.x >= 0 and l.x < self.width and l.y >= 0 and l.y < self.height

    def get_location(self, loc, direction):
        l = copy.deepcopy(loc)
        if direction != STILL:
            if direction == NORTH:
                if l.y == 0:
                    l.y = self.height - 1
                else:
                    l.y -= 1
            elif direction == EAST:
                if l.x == self.width - 1:
                    l.x = 0
                else:
                    l.x += 1
            elif direction == SOUTH:
                if l.y == self.height - 1:
                    l.y = 0
                else:
                    l.y += 1
            elif direction == WEST:
                if l.x == 0:
                    l.x = self.width - 1
                else:
                    l.x -= 1
        return l

    def get_site(self, l, direction=STILL):
        l = self.get_location(l, direction)
        return self.contents[l.y][l.x]
