from models import PINN, QRes, FLS, KAN, PINNsFormer, PINNsFormer_Enc_Only,SetPINN


def get_model(args):
    model_dict = {
        'PINN': PINN,
        'QRes': QRes,
        'FLS': FLS,
        'KAN': KAN,
        'PINNsFormer': PINNsFormer,
        'PINNsFormer_Enc_Only': PINNsFormer_Enc_Only, # more efficient and with better performance than original PINNsFormer
        'SetPINN': SetPINN,
    }
    return model_dict[args.model]