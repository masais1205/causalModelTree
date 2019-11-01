from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
pandas2ri.activate()

def import_R_library(rpacknames):
    from rpy2.robjects.vectors import StrVector
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    names_to_install = [x for x in rpacknames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))


def learnPC_R(data, attr_label, alpha=.05):
    pcalg = rpackages.importr('pcalg')
    X = data.drop(columns=attr_label)
    y = data[attr_label]
    if len(data) == 0:
        return []
    pcs = pcalg.pcSelect(y, X, alpha).rx2("G")
    return [name for x, name in zip(pcs, pcs.names) if x == True]