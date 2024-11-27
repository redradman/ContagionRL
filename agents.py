class Human:
    id_counter = 1
    def __init__(self, x, y, npi_adherence = 0.5):
        self.id = Human.id_counter
        Human.id_counter += 1

        self.x = x
        self.y = y

class Virus:
    id_counter = 1
    def __init__(self, x, y):
        self.id = Virus.id_counter
        Virus.id_counter += 1

        self.x = x
        self.y = y
        self.is_inside_human = False


