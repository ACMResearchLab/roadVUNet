from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.util import adjust_support
from edflow.util import PRNGMixin
import os
import numpy as np

from VUNet.data.stickman import kp2stick
from VUNet.data.keypoint_models import OPENPOSE_18

from PIL import Image


class Prjoti(MetaDataset):
    def __init__(self, config):
        root = config["data_root"]

        if not os.path.exists(os.path.join(root, 'meta.yaml')):
            print("fix root!")

        super().__init__(root)


class Prjoti_VUNet(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        print(os.getcwd())
        self.base = Prjoti(config)
        self.labels = self.base.labels

        self.root = config["data_root"]

    def get_example(self, idx):
        road = os.listdir(self.root + "/skeleton")[idx][0:-4]
        #skeleton_path = os.path.join(self.root + "/skeleton", os.listdir(self.root + "/skeleton")[idx])
        appearance_path = os.path.join(self.root + "/style", os.listdir(self.root + "/style")[0])
        #target_path = os.path.join(self.root + "/target", os.listdir(self.root + "/target")[idx])
        skeleton_path = os.path.join(self.root + "/skeleton/", road+".tif")
        target_path = os.path.join(self.root + "/target/", road+".tif")

        skeleton = Image.open(skeleton_path)
        skeleton = np.array(skeleton.resize((256, 256), Image.ANTIALIAS))
        skeleton = np.expand_dims(skeleton,axis=2)
        #skeleton = np.concatenate((skeleton,skeleton,skeleton),axis=2)
        skeleton = adjust_support(skeleton, '-1->1', '0->255') # needed

        appearance = np.array(Image.open(appearance_path).convert('RGB'))
        #appearance = np.array(appearance.resize((256, 256), Image.ANTIALIAS))
        appearance = adjust_support(appearance, '-1->1', '0->255') # needed

        target = np.array(Image.open(target_path).convert('RGB'))
        #target = np.array(target.resize((256, 256), Image.ANTIALIAS))
        target = adjust_support(target, '-1->1', '0->255') # needed
       
        return {'stickman': skeleton, 'appearance': appearance, 'target': target}

    def __len__(self):
        return len(os.listdir(self.root + "/skeleton"))


class Prjoti_VUNet_train(DatasetMixin):
    def __init__(self, config):
        self.P = Prjoti_VUNet(config)
        self.data = SubDataset(self.P, np.arange(0, int(0.9 * len(self.P))))


class Prjoti_VUNet_val(DatasetMixin):
    def __init__(self, config):
        self.P = Prjoti_VUNet(config)
        self.data = SubDataset(self.P, np.arange(int(0.9 * len(self.P)), len(self.P)))
