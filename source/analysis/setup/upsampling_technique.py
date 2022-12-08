from enum import Enum

class UpsamplingTechnique(Enum):
    none = "No Upsampling Performed"
    random_duplication = "Random Duplication of Data Points"
    SMOTE = "SMOTE Upsampling Methode"
    