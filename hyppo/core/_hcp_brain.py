from owlready2 import *

hcp_brain_onto = get_ontology("http://synthesis.ipi.ac.ru/hcp_onto.owl")

with hcp_brain_onto:
    class Human(Thing): pass
    class Brain(Thing): pass
    class Image(Thing): pass
    class ROI(Thing): pass
    class Voxel(Thing): pass
    class Atlas(Thing): pass
    class RoiVoxelMapping(Thing): pass

    class has_for_age_from(Human >> int, DataProperty, FunctionalProperty): pass
    class has_for_age_to(Human >> int, DataProperty, FunctionalProperty): pass

    class has_for_gender(Human >> str, DataProperty, FunctionalProperty): pass
    class has_for_brain(Human >> Brain): pass
    class has_for_image(Brain >> Image): pass
    class has_for_voxel(Image >> Voxel): pass


    class has_for_roi_voxel_mapping(Atlas >> RoiVoxelMapping): class_property_type = ["some"]
    class has_for_roi(RoiVoxelMapping >> ROI): class_property_type = ["exactly"]
    class has_for_voxel(RoiVoxelMapping >> Voxel): class_property_type = ["exactly"]



    class Adult(Human):
        equivalent_to = [Human & has_for_age_from > 18]

    class YoungAdult(Adult):
        equivalent_to = [Adult & has_for_age_to < 30]

    class MiddleAgedAdult(Adult):
        equivalent_to = [Adult & has_for_age_from > 30 & has_for_age_to < 40]

    class Man(Human):
        equivalent_to = [Human & has_for_gender == 'male']

    class Woman(Human):
        equivalent_to = [Human & has_for_gender == 'woman']

    AllDisjoint([Man, Woman])