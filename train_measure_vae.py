import click

from MeasureVAE.measure_vae import MeasureVAE
from MeasureVAE.vae_trainer import VAETrainer
from MeasureVAE.vae_tester import VAETester
from data.dataloaders.bar_dataset import *
from utils.helpers import *
import torch
import tqdm

import os, shutil, heapq

from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import gaussian_kde
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"]="1"

@click.command()
@click.option('--note_embedding_dim', default=10,
              help='size of the note embeddings')
@click.option('--metadata_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_encoder_layers', default=2,
              help='number of layers in encoder RNN')
@click.option('--encoder_hidden_size', default=512,
              help='hidden size of the encoder RNN')
@click.option('--encoder_dropout_prob', default=0.5,
              help='float, amount of dropout prob between encoder RNN layers')
@click.option('--has_metadata', default=False,
              help='bool, True if data contains metadata')
@click.option('--latent_space_dim', default=256,
              help='int, dimension of latent space parameters')
@click.option('--num_decoder_layers', default=2,
              help='int, number of layers in decoder RNN')
@click.option('--decoder_hidden_size', default=512,
              help='int, hidden size of the decoder RNN')
@click.option('--decoder_dropout_prob', default=0.5,
              help='float, amount got dropout prob between decoder RNN layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=16, # 16 EPOCHS
              help='number of training epochs')
@click.option('--train/--test', default=False, # TRAIN
              help='train or test the specified model')
@click.option('--plot/--no_plot', default=True,
              help='plot the training log')
@click.option('--log/--no_log', default=True,
              help='log the results for tensorboard')
@click.option('--reg_loss/--no_reg_loss', default=True, # YES REG LOSS
              help='train with regularization loss')
@click.option('--reg_type', default='four_metrics', # REG TYPE FOUR METRICS
              help='attribute name string to be used for regularization')
@click.option('--reg_dim', default=0, # REG DIMS, overwritten in vae_trainer
              help='dimension along with regularization is to be carried out')
@click.option('--attr_plot/--no_attr_plot', default=True,
              help='if True plots the attribute dsitributions, else produces interpolations')
def main(note_embedding_dim,
         metadata_embedding_dim,
         num_encoder_layers,
         encoder_hidden_size,
         encoder_dropout_prob,
         latent_space_dim,
         num_decoder_layers,
         decoder_hidden_size,
         decoder_dropout_prob,
         has_metadata,
         batch_size,
         num_epochs,
         train,
         plot,
         log,
         reg_loss,
         reg_type,
         reg_dim,
         attr_plot
         ):

    is_short = False
    num_bars = 1
    folk_dataset_train = FolkNBarDataset(
        dataset_type='train',
        is_short=is_short,
        num_bars=num_bars)
    folk_dataset_test = FolkNBarDataset(
        dataset_type='test',
        is_short=is_short,
        num_bars=num_bars
    )

    model = MeasureVAE(
        dataset=folk_dataset_train,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_encoder_layers=num_encoder_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_dropout_prob=encoder_dropout_prob,
        latent_space_dim=latent_space_dim,
        num_decoder_layers=num_decoder_layers,
        decoder_hidden_size=decoder_hidden_size,
        decoder_dropout_prob=decoder_dropout_prob,
        has_metadata=has_metadata
    )

    if train:
        if torch.cuda.is_available():
            model.cuda()
        trainer = VAETrainer(
            dataset=folk_dataset_train,
            model=model,
            lr=1e-4,
            has_reg_loss=reg_loss,
            reg_type=reg_type,
            reg_dim=reg_dim
        )
        trainer.train_model(
            batch_size=batch_size,
            num_epochs=num_epochs,
            plot=plot,
            log=log,
        )

# **************************************************  Eval  **************************************************************************************        
    else:

        model.load()
        model.cuda()
        model.eval()

        folk_dataset_train.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )

        (generator_train,
         generator_val,
         generator_test) = folk_dataset_train.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )

        # ******************************** Z Vectors *************************************************

        z0_actual_values = []
        z1_actual_values = []
        z2_actual_values = []
        z3_actual_values = []

        for sample_id, (score_tensor, metadata_tensor) in enumerate(generator_train):

            if isinstance(folk_dataset_train, FolkNBarDataset):
                batch_size = score_tensor.size(0)
                score_tensor = score_tensor.view(batch_size, folk_dataset_train.n_bars, -1)
                score_tensor = score_tensor.view(batch_size * folk_dataset_train.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size, folk_dataset_train.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size * folk_dataset_train.n_bars, -1)

            # convert input to torch Variables
            score_tensor, metadata_tensor = (
                to_cuda_variable_long(score_tensor),
                to_cuda_variable_long(metadata_tensor)
            )

            # compute encoder forward pass
            z_dist = model.encoder(score_tensor)
            # sample from distribution
            z_tilde = z_dist.rsample()

            for z_tilde_index in range(z_tilde.size()[0]):
                dim_0_val = z_tilde[z_tilde_index, 0].cpu().detach().numpy()
                dim_1_val = z_tilde[z_tilde_index, 1].cpu().detach().numpy()
                dim_2_val = z_tilde[z_tilde_index, 2].cpu().detach().numpy()
                dim_3_val = z_tilde[z_tilde_index, 3].cpu().detach().numpy()

                z0_actual_values.append(dim_0_val.tolist())
                z1_actual_values.append(dim_1_val.tolist())
                z2_actual_values.append(dim_2_val.tolist())
                z3_actual_values.append(dim_3_val.tolist())

        # ************************************ For Regularised Latent Dimensions - Value Histograms *********************************************************

        
        num_of_bins = 100
        plt.figure()
        plt.hist(z0_actual_values, bins = 100)
        plt.savefig('z0_hist.png')
        plt.figure()
        plt.hist(z1_actual_values, bins=num_of_bins)
        plt.savefig('z1_hist.png')
        plt.figure()
        plt.hist(z2_actual_values, bins=num_of_bins)
        plt.savefig('z2_hist.png')
        plt.figure()
        plt.hist(z3_actual_values, bins=num_of_bins)
        plt.savefig('z3_hist.png')

        # ************************** Data Contribution to the Latent Space Formation - 2D Plots for Metric Combinations ************************************

        num_of_samples = 10

        rhy_complx_min_val = -1.8
        rhy_complx_max_val = 2.2

        avg_int_jump_min_val = -4.0
        avg_int_jump_max_val = 10.0

        dim_rhy_complx_values = np.linspace(rhy_complx_min_val, rhy_complx_max_val, num = num_of_samples) # 0

        dim_pitch_range_values = [-1.0090625762939454, -0.8182210624217987, -0.5892112457752228, -0.36020142912864683, -0.16935991525650018, 0.059649901390075755, 0.28865971803665147, 0.5176695346832276, 0.7085110485553741, 0.8993525624275205]
        
        dim_note_density_values = [-1.6128587484359742, -1.3588566422462462, -0.596850323677063, -0.19044695377349852, 0.16515599489212018, 0.5207589435577393, 0.9271623134613036, 1.2827652621269228, 1.5875677895545959, 1.8415698957443238]

        dim_avg_int_jump_values = np.linspace(avg_int_jump_min_val, avg_int_jump_max_val, num = num_of_samples)
        
        rhy_complx_scale_coeff = 3.0
        pitch_range_scale_coeff = 6.0
        note_density_scale_coeff = 1.75
        avg_int_jump_scale_coeff = 1.0

        num_of_bins = 25
        gamma = 0.12

        x_low_limit = rhy_complx_min_val * rhy_complx_scale_coeff
        x_high_limit = rhy_complx_max_val * rhy_complx_scale_coeff

        y_low_limit = dim_pitch_range_values[0] * pitch_range_scale_coeff
        y_high_limit = dim_pitch_range_values[9] * pitch_range_scale_coeff

        H, xedges, yedges = np.histogram2d(z0_actual_values, z1_actual_values, bins = num_of_bins, range = [[x_low_limit, x_high_limit], [y_low_limit, y_high_limit]])
        H = H.T

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect = 'auto', xlim = xedges[[0, -1]], ylim = yedges[[0, -1]])
        im = NonUniformImage(ax, interpolation = 'bilinear', norm = mcolors.PowerNorm(gamma))
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        im.set_data(xcenters, ycenters, H)
        ax.images.append(im)
       
        plt.savefig('RC_vs_PR_Data_in_Latent_2D_Plot.png')

        # ********* 

        x_low_limit = dim_note_density_values[0] * note_density_scale_coeff
        x_high_limit = dim_note_density_values[9] * note_density_scale_coeff

        y_low_limit = avg_int_jump_min_val * avg_int_jump_scale_coeff
        y_high_limit = avg_int_jump_max_val * avg_int_jump_scale_coeff

        H, xedges, yedges = np.histogram2d(z2_actual_values, z3_actual_values, bins = num_of_bins, range = [[x_low_limit, x_high_limit], [y_low_limit, y_high_limit]])
        H = H.T

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect = 'auto', xlim = xedges[[0, -1]], ylim = yedges[[0, -1]])
        im = NonUniformImage(ax, interpolation = 'bilinear', norm = mcolors.PowerNorm(gamma))
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        im.set_data(xcenters, ycenters, H)
        ax.images.append(im)

        plt.savefig('ND_vs_AIJ_Data_in_Latent_2D_Plot.png')

        # ************************************************ Generate MIDI Files *****************************************************************

        midis_path = os.getcwd() + '/generated_midi_files/'

        if not os.path.exists(midis_path):
            os.makedirs(midis_path)


        original_midi_file_path = os.getcwd() + '/input_midi.mid'
        # original_abc_file_path = original_midi_file_path[:-4] + '.abc'
        s = music21.converter.parse(original_midi_file_path)
        # s.write('midi', fp = os.getcwd() + '/midi_files_metrics_final_withLSR_ver2/input_midi_COOL_INPUT_RECON.mid')

        
        # original_abc_score = get_music21_score_from_path(original_abc_file_path)
        # original_abc_score_tensor = folk_dataset_train.get_tensor(original_abc_score)
        original_abc_score_tensor = folk_dataset_train.get_tensor(s)
        z_dist = model.encoder(original_abc_score_tensor.cuda())
        # sample from distribution
        z_tilde = z_dist.rsample()

        z_original = z_tilde[0]

        measure_seq_len = 24 # hard coded, taken from vae_tester
        train = False
        batch_size_inference = 1 # hard coded, taken from vae_tester

        # midi save original source midi

        z_original = z_original.unsqueeze(0)

        dummy_score_tensor = to_cuda_variable(torch.zeros(batch_size_inference, measure_seq_len))
        _, sam1_original = model.decoder(z_original, dummy_score_tensor, train)

        sam1_score_original = folk_dataset_train.get_score_from_tensor(sam1_original.cpu())

        sam1_score_original.write('midi', os.getcwd() + '/generated_midi_files/input_midi.mid')

        midi_file_counter = 0
        for rhy_complx_index in range(num_of_samples):
            for pitch_range_index in range(num_of_samples):
                for note_density_index in range(num_of_samples):
                    for avg_int_jump_index in range(num_of_samples):

                        midi_file_counter += 1

                        print('Processing: ' + str(midi_file_counter))

                        z = z_tilde[0]
                        z[0] = dim_rhy_complx_values[rhy_complx_index] * rhy_complx_scale_coeff# 0
                        z[1] = dim_pitch_range_values[pitch_range_index] * pitch_range_scale_coeff# 1 or 2
                        z[2] = dim_note_density_values[note_density_index] * note_density_scale_coeff
                        z[3] = dim_avg_int_jump_values[avg_int_jump_index] * avg_int_jump_scale_coeff

                        z = z.unsqueeze(0)

                        dummy_score_tensor = to_cuda_variable(torch.zeros(batch_size_inference, measure_seq_len))
                        _, sam1 = model.decoder(z, dummy_score_tensor, train)

                        sam1_score = folk_dataset_train.get_score_from_tensor(sam1.cpu())

                        midi_file_name = 'midi_' + str(rhy_complx_index + 1) + '_' + str(pitch_range_index + 1) + '_' + str(note_density_index + 1) + '_' + str(avg_int_jump_index + 1) + '.mid'

                        sam1_score.write('midi', os.getcwd() + '/generated_midi_files/' + midi_file_name)

        # ************************************ For 2D Plots - Surface Maps *********************************************************

        tester = VAETester(
            dataset=folk_dataset_test,
            model=model,
            has_reg_loss=reg_loss,
            reg_type=reg_type,
            reg_dim=reg_dim
        )

        # dim, score = tester.test_interpretability(
        #     batch_size=batch_size,
        #     attr_type='note_range'
        # )

        grid_res = 0.05
        tester.plot_attribute_surface(
               dim1=2,
               dim2=3,
               grid_res=grid_res,
               x_min = x_low_limit,
               x_max = x_high_limit,
               y_min = y_low_limit,
               y_max = y_high_limit,
               z_source = z_original,
           )


        # tester.test_interp()
        # tester.plot_transposition_points(plt_type='tsne')
        # if attr_plot:
        #    grid_res = 0.05
        #    tester.plot_data_attr_dist(
        #        dim1=0,
        #       dim2=2,
        #    )
        #    tester.plot_attribute_surface(
        #        dim1=0,
        #        dim2=2,
        #        grid_res=grid_res
        #    )
        # tester.plot_attribute_surface(
        #    dim1=29,
        #    dim2=241,
        #    grid_res=grid_res
        # )
        # else:
        #    tester.test_attr_reg_interpolations(
        #        dim=1,
        #    )


if __name__ == '__main__':
    main()