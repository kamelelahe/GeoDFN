class bufferZoneCalculator:
    def __init__(self, bufferZone):
        self.constant=bufferZone['constant']
        self.method=bufferZone['method']
    def calculate(self, fractures):
        if self.method == 'constant':
            for fracture in fractures:
                fracture['fracture spacing'] = self.constant
            return fractures
        elif self.method=='linearRelationshipLength':
            for fracture in fractures:
                # usually considered to be 10% of fracture length
                fracture['fracture spacing'] = self.constant * fracture['fracture length']
            return fractures

