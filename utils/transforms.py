import numpy as np


class Transforms():
    def __init__(self, nb_epi, actions):
        self.nb_epi = nb_epi
        self.actions = actions

    def continuous_2_descrete(self, obj, obs, groups):
        """
            Convert Observation to Discrete
        :param obs: env state
        :return: transformed obs
        """

        location = obj
        for i in range(len(obs)):
            value = int(obs[i] * 100) % groups
            location = location[value]
        return location

    def continuous_2_dict(self, obj, obs):
        """
            Locate obs in dictionary, if n/a create reward group
        Parameters
        ----------
        obs: env state

        Returns
        -------
        location in dictionary

        """
        location = obj
        if not isinstance(obs, str):
            value = ""
            for i in range(len(obs)):
                value += str(round(obs[i] * np.log(self.nb_epi), 0))
        else: value = obs
        try:
            location = location[value]
        except KeyError:
            location[value] = [0 for i in range(self.actions)]
        return value