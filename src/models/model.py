import src.models.model_wave_simsiam as model_wave_simsiam
import src.models.model_downstream as model_downstream
import torch


def load_model(config, model_name, checkpoint_path=None, pretext_model=None):
    model = None
    if model_name == "WaveSimSiam":
        model = model_wave_simsiam.WaveSimSiam(
            config=config,
            encoder_input_dim=config['encoder_input_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            encoder_filter_size=config['encoder_filter_size'],
            encoder_stride=config['encoder_stride'],
            encoder_padding=config['encoder_padding'],
            mlp_input_dim=config['mlp_input_dim'],
            mlp_hidden_dim=config['mlp_hidden_dim'],
            mlp_output_dim=config['mlp_output_dim'],
            feature_extractor_model=config['feature_extractor_model'],
            pretrain=config['feature_extractor_model_pretrain'],
            dropout=config['encoder_dropout']
        )
    elif model_name == "DownstreamFlatClassification":
        model = model_downstream.DownstreamFlatClassification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )
    elif model_name == "DownstreamLinearClassification":
        model = model_downstream.DownstreamLinearClassification(
            input_dim=config['downstream_input_dim'],
            hidden_dim=config['downstream_hidden_dim'],
            output_dim=config['downstream_output_dim'],
        )


    if checkpoint_path is not None:
        print("load checkpoint...")
        print(checkpoint_path)
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model
