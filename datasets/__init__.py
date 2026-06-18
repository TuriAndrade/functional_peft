from .vtab_caltech101 import VTABCaltech101Dataset
from .vtab_cifar100 import VTABCIFAR100Dataset
from .vtab_dtd import VTABDTDDataset
from .vtab_flowers102 import VTABFlowers102Dataset
from .vtab_pets import VTABPetsDataset
from .vtab_sun397 import VTABSun397Dataset
from .vtab_svhn import VTABSVHNDataset
from .vtab_eurosat import VTABEuroSATDataset
from .vtab_resisc45 import VTABRESISC45Dataset
from .vtab_patch_camelyon import VTABPatchCamelyonDataset
from .vtab_retinopathy import VTABRetinopathyDataset
from .vtab_clevr_count import VTABCLEVRCountDataset
from .vtab_clevr_distance import VTABCLEVRDistanceDataset
from .vtab_dmlab import VTABDMLabDataset
from .vtab_kitti_distance import VTABKITTIDistanceDataset
from .vtab_dsprites_location import VTABDSpritesLocationDataset
from .vtab_dsprites_orientation import VTABDSpritesOrientationDataset
from .vtab_smallnorb_azimuth import VTABSmallNORBAzimuthDataset
from .vtab_smallnorb_elevation import VTABSmallNORBElevationDataset
from .fgvc_cub200 import FGVCCUB200Dataset
from .fgvc_nabirds import FGVCNABirdsDataset
from .fgvc_oxford_flowers import FGVCOxfordFlowersDataset
from .fgvc_stanford_dogs import FGVCStanfordDogsDataset
from .fgvc_stanford_cars import FGVCStanfordCarsDataset
from .imagenet_val import ImageNetValDataset

datasets = {
    "vtab_caltech101": VTABCaltech101Dataset,
    "vtab_cifar100": VTABCIFAR100Dataset,
    "vtab_dtd": VTABDTDDataset,
    "vtab_flowers102": VTABFlowers102Dataset,
    "vtab_pets": VTABPetsDataset,
    "vtab_sun397": VTABSun397Dataset,
    "vtab_svhn": VTABSVHNDataset,
    "vtab_eurosat": VTABEuroSATDataset,
    "vtab_resisc45": VTABRESISC45Dataset,
    "vtab_patch_camelyon": VTABPatchCamelyonDataset,
    "vtab_retinopathy": VTABRetinopathyDataset,
    "vtab_clevr_count": VTABCLEVRCountDataset,
    "vtab_clevr_distance": VTABCLEVRDistanceDataset,
    "vtab_dmlab": VTABDMLabDataset,
    "vtab_kitti_distance": VTABKITTIDistanceDataset,
    "vtab_dsprites_location": VTABDSpritesLocationDataset,
    "vtab_dsprites_orientation": VTABDSpritesOrientationDataset,
    "vtab_smallnorb_azimuth": VTABSmallNORBAzimuthDataset,
    "vtab_smallnorb_elevation": VTABSmallNORBElevationDataset,
    "fgvc_cub200": FGVCCUB200Dataset,
    "fgvc_nabirds": FGVCNABirdsDataset,
    "fgvc_oxford_flowers": FGVCOxfordFlowersDataset,
    "fgvc_stanford_dogs": FGVCStanfordDogsDataset,
    "fgvc_stanford_cars": FGVCStanfordCarsDataset,
    "imagenet_val": ImageNetValDataset,
}
