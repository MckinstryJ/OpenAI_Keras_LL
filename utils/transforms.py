def continuous_2_descrete(obj, obs, groups):
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


def continuous_2_dict(obj, obs, nb_epi, action_dim):
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
    value = ""
    for i in range(len(obs)):
        value += str(round(obs[i] * (nb_epi / 1000.), 0))
    try:
        location = location[value]
    except KeyError:
        location[value] = [0 for i in range(action_dim)]
    return value