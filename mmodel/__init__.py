
def get_module(name):
    if name == 'TFDN':
        from .TFDN import TFDN_model
        return TFDN_model.params, TFDN_model.TFDN()