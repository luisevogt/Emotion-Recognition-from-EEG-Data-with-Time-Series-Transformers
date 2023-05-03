import yaml


class Config:
    def __init__(self):
        self.config_dict = {}

    def get_args(self, path):
        """
        Reads hyperparameters of specified .yaml file and returns dictionary with arguments.
        :param path: The .yaml file path.
        :return: A dictionary containing all arguments.
        """

        with open(path, "r") as stream:
            args = yaml.safe_load(stream)

        self.config_dict = args

        return self.config_dict

    @staticmethod
    def store_args(path, args):
        with open(path, "w") as stream:
            yaml.safe_dump(args, stream)
