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
            num_to_evacuate = min(len(self.inventory), random.randint(1,3))
            for _ in range(num_to_evacuate):
                idx = random.randint(0,len(self.inventory)-1)
                blob = self.inventory.pop(idx)
                blob.is_hidden = False

                offset_x = random.uniform(-6.0, 6.0)
                offset_y = random.uniform(-6.0, 6.0)

                blob.x = self.door_coords[0] + offset_x
                blob.y = self.door_coords[1] + offset_y
                blob.start_x, blob.start_y = blob.x, blob.y
                blob.end_x, blob.end_y = blob.x, blob.y

                blob.become_panicked()