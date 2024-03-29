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
    class Connectivity(Thing): pass
    class FunctionalConncectivity(Connectivity): pass

    class has_for_age(Human >> int, DataProperty, FunctionalProperty): pass
    class has_for_condition(Human >> str, DataProperty): pass

    class has_for_gender(Human >> str, DataProperty, FunctionalProperty): pass
    class has_for_brain(Human >> Brain): pass
    class has_for_image(Brain >> Image): pass
    class has_for_connectivity(Brain >> Connectivity): pass
    class has_for_voxel(Image >> Voxel): pass


    class has_for_roi_voxel_mapping(Atlas >> RoiVoxelMapping): class_property_type = ["some"]
    class has_for_roi(RoiVoxelMapping >> ROI): class_property_type = ["exactly"]
    class has_for_voxel(RoiVoxelMapping >> Voxel): class_property_type = ["exactly"]

    class fMRI(Image): pass
    class has_for_x(Voxel >> int, DataProperty, FunctionalProperty): pass
    class has_for_y(Voxel >> int, DataProperty, FunctionalProperty): pass
    class has_for_z(Voxel >> int, DataProperty, FunctionalProperty): pass
    class has_for_t(Voxel >> float, DataProperty, FunctionalProperty): pass
    class has_for_color(Voxel >> int, DataProperty, FunctionalProperty): pass


    class rsfMRI(fMRI): pass

    class Adult(Human):
        equivalent_to = [Human & has_for_age > 18]

    class YoungAdult(Adult):
        equivalent_to = [Adult & has_for_age < 30]

    class MiddleAgedAdult(Adult):
        equivalent_to = [Adult & has_for_age > 30 & has_for_age < 45]

    class Man(Human):
        equivalent_to = [Human & has_for_gender == 'male']

    class Woman(Human):
        equivalent_to = [Human & has_for_gender == 'woman']

    AllDisjoint([Man, Woman])
    AllDisjoint([MiddleAgedAdult, YoungAdult])