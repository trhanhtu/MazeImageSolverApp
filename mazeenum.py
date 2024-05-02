from enum import IntEnum
 
class Maze(IntEnum):
    EMPTY = 1
    PATH = 2
    START = 3
    END = 4
    WALL = 5
    WIDTH = 39
    HEIGHT = 39