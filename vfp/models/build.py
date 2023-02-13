from .binary_head import RBE_NORM


def build_binary_head(config, in_planes):
    model_type = config.TRANS.TYPE
    if model_type == 'rbe_norm':
        model = RBE_NORM(in_planes,
                  config.TRANS.RBE.OUTPUT_DIM,
                  num_layers=config.TRANS.RBE.NUM_LAYERS,
                  hidden_dim=config.TRANS.RBE.HIDDEN_DIM,
                  bias=config.TRANS.RBE.BIAS,
                  binary_func=config.TRANS.RBE.BINARY_FUNC,
                  transform_blocks=config.TRANS.RBE.TRANSFORM_BLOCKS,
                  proxy_loss=config.TRAIN.PROXY_LOSS)
        num_features = config.TRANS.RBE.OUTPUT_DIM
    else:
        raise NotImplementedError(f"Unkown binary head: {model_type}")

    return model, num_features
