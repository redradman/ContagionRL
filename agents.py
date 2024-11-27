
class Human:
    id_counter = 1
    def __init__(self, x, y, npi_adherence = 0.5, is_infected = False):
        self.id = Human.id_counter
        Human.id_counter += 1

        self.x = x
        self.y = y
        self.state = "I" if is_infected else "S"
