
from .data_generator import DataGenerator, random


class AddPayFieldGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.output = dict()

    def return_synthetic_data(self):
        while True:
            self.output = dict()
            lith_list = self.lithologies()
            # PickLith
            self.output['lithology'] = lith_list[random.randint(0, 3)]

            # Assign Contacts
            self.output['FWL'] = random.randint(-10000, 2000)
            self.output['GOC'] = random.randint(self.output['FWL'], 2100)
            # PoroPerm Transform
            A = random.uniform(0.8,
                               1.1)  # TODO: Modify this based on lithology
            M = random.uniform(1.8, 2.2)

            # JFUNC
            self.output['J_A'] = 6.575
            self.output['J_B'] = -0.4075
            self.output['J_C'] = 0.04
            # Viscosity
            self.output['viscosity'] = random.uniform(1.0, 200.0)
            # Expected Recovery
            self.output['expected_recovery'] = 0.45
            # Max Porosity
            self.output['max_porosity'] = 0.34
            self.output['hperm'], self.output['jperm'] = self.get_perm_transform_params(
                self.output['lithology'])
            self.output['has_aquifer'] = random.randint(0, 1)
            self.output['VRR'] = random.uniform(0.0, 5.0)
            self.output['discovery_pressure'] = 2000 + (
                        random.randint(-1000, 1500) * (
                        1 + (2000 - self.output['FWL']) / 12000))
            self.output['GOR'] = random.randint(0, 4000)
            yield self.output

    @staticmethod
    def lithologies():
        return ['sandstone', 'mixed_porc_sand', 'porc', 'unconsolidated_sand']

    @staticmethod
    def get_perm_transform_params(lithology):
        if lithology == 'sandstone':
            hperm = random.randint(10, 40)
            jperm = random.uniform(-3.5, -3.0)
        elif lithology == 'mixed_porc_sand':
            hperm = random.randint(10, 25)
            jperm = random.uniform(-3.0, -2.5)
        elif lithology == 'porc':
            hperm = random.randint(8, 15)
            jperm = random.uniform(-3.0, -2.5)
        elif lithology == 'unconsolidated_sand':
            hperm = random.randint(20, 50)
            jperm = random.uniform(-3.5, -3.4)
        return hperm, jperm
