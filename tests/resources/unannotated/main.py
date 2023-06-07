import os.path

import pandas as pd

class CSVParser:
    SUFFIX = "tests"
    def __init__(self, root):
        """
        Args:
            root (str): String referring to path of folder to search in

        Returns:
            CSVParser: Initialised CSVParser at given folder root
        """
        self.root = root
        
    def parse(self, filename):
        """
        Args:
            filename (str): filename within root / SUFFIX

        Returns: a CSV file, parsed by pandas and its underlying engine

        """
        csv_contents = pd.read_csv(os.path.join(self.root, CSVParser.SUFFIX, filename))
        return csv_contents
        
        
if __name__ == "__main__":
    parser = CSVParser(os.getcwd())
    document = parser.parse("inferred.csv")

    print(document)

