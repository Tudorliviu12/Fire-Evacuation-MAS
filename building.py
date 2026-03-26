import random

class Building:
    def __init__(self, name, door_node, door_coords, area):
        self.name = name
        self.door_node = door_node
        self.door_coords = door_coords
        self.inventory = []
        self.is_on_fire = False
        self.area = area

    def evacuate_step(self):
        if self.is_on_fire and len(self.inventory) > 0:
            num_to_evacuate = min(len(self.inventory), random.randint(1,2))
            for _ in range(num_to_evacuate):
                blob = self.inventory.pop(0)
                blob.is_hidden = False

                blob.x, blob.y = self.door_coords
                blob.start_x, blob.start_y = self.door_coords
                blob.end_x, blob.end_y = self.door_coords
                blob.become_panicked()

