
from moleculib.abstract.dataset import PreProcessedDataset
import os 

class D3PM(PreProcessedDataset):


    @classmethod
    def build():
        pass

if __name__ == '__main__':
    base_path = './'
    D3PM.build(base_path)