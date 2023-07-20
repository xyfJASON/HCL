from engine.tester import Tester
from utils.data import get_dataset


class DownstreamTester(Tester):
    def create_data(self):
        test_set = get_dataset(
            name=self.cfg.data.name,
            dataroot=self.cfg.data.dataroot,
            img_size=self.cfg.data.img_size,
            split='test',
        )
        return test_set
