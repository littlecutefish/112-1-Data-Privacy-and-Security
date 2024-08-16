from .basic_mondrian import basic_mondrian_anonymize, read_tree, mondrian_ldiv_anonymize

from utils.types import AnonMethod

def k_anonymize(anon_params):

    if anon_params["name"] == AnonMethod.BASIC_MONDRIAN:
        return basic_mondrian_anonymize(
            anon_params["value"],
            anon_params["att_trees"],
            anon_params["data"],
            anon_params["qi_index"],
            anon_params["sa_index"])

    if anon_params["name"] == AnonMethod.MONDRIAN_LDIV:
        return mondrian_ldiv_anonymize(
            anon_params["value"],
            anon_params["att_trees"],
            anon_params["data"],
            anon_params["qi_index"],
            anon_params["sa_index"])
