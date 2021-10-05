from ._base import Artefact

class VirtualExperiment(Artefact):

    def __init__(self):
        pass

    def run(self):
        pass

    def save(self):
        pass

    def load(db, type, id):
        db.connect()
        try:
            print(f'Loading from: {cls.root / storage / filename}') if cls.debug else None
            with open(cls.root / storage / filename, 'rb') as f:
                return pickle.load(f)

            if type == 'hypotheses':
                artefact = db._load_hypothesis(id)
            elif type == 'model':
                artefact = db._load_model(id)
            elif type == 'ontology':
                artefact = db._load_ontology(id)
            elif type == 'workflow':
                artefact = db._load_workflow(id)
            elif type == 'lattice':
                artefact = db._load_lattice(id)

            return artefact
        except FileNotFoundError:
            print('Объект отсутствует в базе данных.')

    def add(self, type, artefact):
        if type == 'hypotheses':
            self.hypotheses.append(artefact)
        elif type == 'model':
            self.model.append(artefact)
        elif type == 'ontology':
            self.ontology.append(artefact)
        elif type == 'workflow':
            self.workflow = artefact
        elif type == 'lattice':
            self.lattice.append(artefact)

    def modify(self, type, artefact):

        if type == 'hypotheses':
            artefacts = []

            for i in range(1, len(H_list)):
                H1 = H_list[i][0]
                H2 = H_list[i][1]
                print("Check pair: ", (H1, H2))

                connected_h, is_connected = concat_H(data.copy(), H1, H2)
                if is_connected:
                    artefacts.append((H1, H2))

        elif type == 'model':
            artefacts = []

            for i in range(1, len(artefact)):
                M1 = artefact[i][0]
                M2 = artefact[i][1]
                print("Check pair: ", (M1, M2))

                connected_h, is_connected = concat_H(data.copy(), M1, M2)
                if is_connected:
                    artefacts.append((M1, M2))

        return artefacts









