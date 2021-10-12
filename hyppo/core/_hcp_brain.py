from owlready2 import *

hcp_brain_onto = get_ontology("http://synthesis.ipi.ac.ru/hcp_onto.owl")

with hcp_brain_onto:
    class Human(Thing): pass
    class Image(Thing): pass
    class Brain(Thing): pass

    class has_for_age_from(Human >> int, DataProperty, FunctionalProperty): pass
    class has_for_age_to(Human >> int, DataProperty, FunctionalProperty): pass

    class has_for_gender(Human >> str): pass
    class has_for_brain(Human >> Brain): pass
    class has_for_image(Brain >> Image): pass

    class YoungAdult(Human):
        equivalent_to = [Human & has_for_age_from > 20 & has_for_age_to < 30]

    class MiddleAgedAdult(Human):
        equivalent_to = [Human & has_for_age_from > 30 & has_for_age_to < 40]

    class Man(Human):
        equivalent_to = [Human & has_for_gender == 'male']

    class Woman(Human):
        equivalent_to = [Human & has_for_gender == 'woman']