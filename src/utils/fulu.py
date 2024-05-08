MAN = 0
PIN = 1
SO = 2
ZU = 3
SHUNZI = 0
KEZI = 1
KANGZI = 2

class Fulu:
    def __init__(self, type, tile_type, number) -> None:
        self.type = type
        self.tile_type = tile_type
        self.number = number

    def __init__(self, tile_str) -> None:
        self.type = type
        self.tile_type = tile_type
        self.number = number

    def __convert(self, tile_str):
        number = ""
        if (tile_str.find('m') != -1):
            tile_str = tile_str.split('m', 1)
            number = tile_str[0]
        elif (tile_str.find('p') != -1):
            tile_str = tile_str.split('p', 1)
            number = tile_str[0]
        elif (tile_str.find('s') != -1):
            tile_str = tile_str.split('s', 1)
            number = tile_str[0]
        elif (tile_str.find('z') != -1):
            tile_str = tile_str.split('z', 1)
            number = tile_str[0]
        return number