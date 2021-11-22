import numpy as np
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss, SupPatchNCELoss
import util.util as util
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.nt_xent import NTXentLoss


class CUTCosegSumModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_PCL', type=float, default=1.0, help='weight for PCL loss: PCL(G(X), X)')
        parser.add_argument('--lambda_GCL', type=float, default=1.0, help='weight for GCL loss: GCL(SegEnc(X), X)')
        parser.add_argument('--pcl_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use PCL loss for identity mapping: PCL(G(Y), Y))')
        parser.add_argument('--pcl_layers', type=str, default='0,4,8,12,16', help='compute PCL loss on which layers')
        parser.add_argument('--pcl_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='label_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--pcl_T', type=float, default=0.07, help='temperature for PCL loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(pcl_idt=True, lambda_PCL=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                pcl_idt=False, lambda_PCL=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'PCL', "GCL"]
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.pcl_layers = [int(i) for i in self.opt.pcl_layers.split(',')]

        if opt.pcl_idt and self.isTrain:
            self.loss_names += ['PCL_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'D_S', 'S_B']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.nt_xent_criterion = NTXentLoss(self.device, opt.temperature, True)
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                      self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netS_B = networks.define_S(opt.input_nc, opt.num_classes, opt.ngf, opt.netS, opt.normG, not opt.no_dropout,
                                        opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                        self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_S = networks.define_D(opt.num_classes, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                            opt.init_type,
                                            opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionPCL = []
            self.criterionSeg = SoftDiceLoss(batch_dice=True, do_bg=False)

            for pcl_layer in self.pcl_layers:
                self.criterionPCL.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_gan, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_S_B_encoder = torch.optim.Adam(self.netS_B.parameters(), lr=opt.lr,
                                                          betas=(opt.beta1, opt.beta2))
            self.optimizer_S_B = torch.optim.Adam(self.netS_B.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_S = torch.optim.Adam(self.netD_S.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.set_requires_grad(self.netG, True)
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def optimize_seg_parameters(self):
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netS_B, True)

        # update S_B
        self.set_requires_grad([self.netD_S, self.netG], False)
        self.forward_seg()
        self.optimizer_S_B.zero_grad()
        self.backward_S()  # calculate gradients for seg_B
        self.optimizer_S_B.step()

        # update D_S
        self.set_requires_grad(self.netS_B, False)
        self.set_requires_grad(self.netD_S, True)
        self.forward_seg()
        self.optimizer_D_S.zero_grad()
        self.backward_D_S()
        self.optimizer_D_S.step()


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'][0].float().to(self.device)
        self.real_B = input['B' if AtoB else 'A'][0].float().to(self.device)
        self.real_A_label = input['segA' if AtoB else 'segB'][0].long().to(self.device)
        self.real_B_label = input['segB' if AtoB else 'segA'][0].long().to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.pcl_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.pcl_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def forward_seg(self):
        self.fake_B = self.netG(self.real_A)  # G_A(A)
        self.seg_B = self.netS_B(self.real_B)  # S_B(B)
        self.fake_seg_B = self.netS_B(self.fake_B)  # S_B(G_A(A))

    def backward_D_S(self):
        # fake_seg_B = self.fake_seg_pool.query(self.fake_seg_B)
        self.loss_D_S = self.backward_D_basic(self.netD_S, self.seg_B.detach(), self.fake_seg_B.detach())
        self.loss_D_S.backward()

    def backward_S(self):
        """ Calculate the loss for segmenter S_B, includes the """
        pred_fake_B = F.softmax(self.fake_seg_B, dim=1)
        pred_real_B = F.softmax(self.seg_B, dim=1)
        self.loss_fake_S = self.criterionSeg(pred_fake_B, self.real_A_label.squeeze(1))
        self.loss_real_S = self.criterionSeg(pred_real_B, self.real_B_label.squeeze(1))
        self.loss_D_S = self.backward_D_basic(self.netD_S, self.seg_B, self.fake_seg_B)
        if self.opt.lambda_GCL > 0.0:
            self.loss_GCL = self.calculate_GCL_loss()
        else:
            self.loss_GCL = 0.0
        self.loss_S_B = self.loss_fake_S + self.loss_D_S + self.loss_GCL  # + F.cross_entropy(self.fake_seg_B, self.real_A_label.squeeze(1))
        self.loss_S_B.backward()

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and PCL loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_PCL > 0.0:
            self.loss_PCL = self.calculate_PCL_loss(self.real_A, self.fake_B)
        else:
            self.loss_PCL, self.loss_PCL_bd = 0.0, 0.0

        if self.opt.pcl_idt and self.opt.lambda_PCL > 0.0:
            self.loss_PCL_Y = self.calculate_PCL_loss(self.real_B, self.idt_B)
            loss_PCL_both = (self.loss_PCL + self.loss_PCL_Y) * 0.5
        else:
            loss_PCL_both = self.loss_PCL

        self.loss_G = self.loss_G_GAN + loss_PCL_both
        return self.loss_G

    def calculate_PCL_loss(self, src, tgt):
        n_layers = len(self.pcl_layers)
        feat_q = self.netG(tgt, self.pcl_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.pcl_layers, encode_only=True)
        feat_k_pool, sample_ids, label_k_pool = self.netF(feat_k, self.real_A_label.float(), self.opt.num_patches, None)
        feat_q_pool, _, _ = self.netF(feat_q, self.real_A_label.float(), self.opt.num_patches, sample_ids)

        total_pcl_loss = 0.0
        for f_q, f_k, label, crit, pcl_layer in zip(feat_q_pool, feat_k_pool, label_k_pool, self.criterionPCL,
                                                    self.pcl_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_PCL
            total_pcl_loss += loss.mean()

        return total_pcl_loss / n_layers

    def calculate_GCL_loss(self):
        # self.fake_B = self.netG(self.real_A)  # G_A(A)
        ft_fake_B = self.netS_B(self.fake_B, encode_only=True)
        ft_A = self.netS_B(self.real_B, encode_only=True)

        ft_fake_B = F.normalize(ft_fake_B, dim=1)
        ft_A = F.normalize(ft_A, dim=1)

        loss = self.nt_xent_criterion(ft_fake_B, ft_A)
        return loss
