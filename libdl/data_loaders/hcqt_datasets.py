import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torchvision import transforms
from scipy.interpolate import interp1d


class dataset_context(torch.utils.data.Dataset):
    """
    Dataset class to be used with DataLoader object. Generates a single HCQT
    frame with context. Note that X (HCQT input) includes the context frames
    but y (pitch (class) target) only refers to the center frame to be predicted.

    Args:
        inputs:         Tensor of HCQT input for one audio file
        targets:        Tensor of pitch (class) targets for the same audio file
        parameters:     Dictionary of parameters with:
        - 'context':        Total number of frames including context frames
        - 'stride':         Hopsize for jumping to the start frame of the next segment
        - 'compression':    Gamma parameter for log compression of HCQT input
        - 'targettype':     'pitch_class' (assumed) or 'pitch'
        - 'aug:transpsemitones':  Data augmentation with transposition (# semitones)
        - 'aug:scalingfactor':    Data augmentation with time scaling (factor)
        - 'aug:randomeq':         Data augmentation with random frequency equalization (amount)
        - 'aug:noisestd':         Data augmentation with random Gaussian noise (standard dev.)
        - 'aug:tuning':           Data augmentation with random tuning shift (+/- 1/3 semitone)
    """
    def __init__(self, inputs, targets, params):
        # Initialization
        torch.initial_seed()
        self.inputs = inputs
        self.targets = targets
        self.context = params['context']
        self.stride = params['stride']
        self.compression = params['compression']
        if 'targettype' not in params:
            params['targettype'] = 'pitch_class'
        self.targettype = params['targettype']
        self.transposition = None
        self.scalingfactor = None
        self.randomeq = None
        self.noisestd = None
        self.tuning = None
        if 'aug:transpsemitones' in params:
            self.transposition = params['aug:transpsemitones']
        if 'aug:scalingfactor' in params:
            self.scalingfactor = params['aug:scalingfactor']
        if 'aug:randomeq' in params:
            self.randomeq = params['aug:randomeq']
        if 'aug:noisestd' in params:
            self.noisestd = params['aug:noisestd']
        if 'aug:tuning' in params:
            self.tuning = params['aug:tuning']

    def __len__(self):
        # Denotes the total number of samples
        return (self.inputs.size()[1]-self.context)//self.stride

    def __getitem__(self, index):
        # Generates one sample of data
        # shift index by half context
        index *= self.stride
        half_context = self.context//2
        index += half_context
        # Load data and get label (remove subharmonic)
        X = self.inputs[:, (index-half_context):(index+half_context+1), :].type(torch.FloatTensor)
        y = torch.unsqueeze(torch.unsqueeze(self.targets[index, :], 0), 1).type(torch.FloatTensor)

        if self.scalingfactor:
            assert False, 'Scaling not implemented for dataset_context!'

        if self.randomeq:
            minval = -1
            while minval<0:
                randomAlpha = torch.randint(1, self.randomeq+1, (1,))
                randomBeta = torch.randint(0, 216, (1,))
                # filtvec = ((1 - (2e-6*randomAlpha*(torch.arange(216)-randomBeta)**2)).unsqueeze(0).unsqueeze(0))
                filtmat = torch.zeros((X.size(0), 1, X.size(2)))
                for nharm in range(filtmat.size(0)):
                    if nharm==0:
                        offset = int(-3*12)
                    else:
                        offset = int(3*12*(np.log2(nharm)))
                    randomBetaHarm = randomBeta - offset
                    currfiltvec = ((1 - (2e-6*randomAlpha*(torch.arange(216)-randomBetaHarm)**2)).unsqueeze(0).unsqueeze(0))
                    filtmat[nharm, :, :] = currfiltvec
                minval = torch.min(filtmat)
            X_filt = filtmat*X
            X = X_filt

        if self.noisestd:
            X += torch.normal(mean=torch.zeros(X.size()), std=self.noisestd*torch.ones(X.size()))
            X_noise = torch.abs(X)
            X = X_noise
            # X_pos = (X>0).type('torch.FloatTensor')

        if self.compression is not None:
            X = np.log(1+self.compression*X)

        if self.tuning:
            tuneshift = torch.randint(-2, 3, (1, )).item()
            tuneshift /= 2.
            X_tuned = X
            if tuneshift==0.5:
                # +0.5:
                X_tuned[:, :, 1:] = (X[:, :, :-1] + X[:, :, 1:])/2
            elif tuneshift==-0.5:
                # -0.5
                X_tuned[:, :, :-1] = (X[:, :, :-1] + X[:, :, 1:])/2
            else:
                X_tuned = torch.roll(X, (int(tuneshift), ), -1)
            if tuneshift>0:
                X_tuned[:, :, :1] = torch.abs(torch.normal(mean=torch.zeros(X_tuned[:, :, :1].size()), std=1e-4*torch.ones(X_tuned[:, :, :1].size())))
            elif tuneshift<0:
                X_tuned[:, :, -1:] = torch.abs(torch.normal(mean=torch.zeros(X_tuned[:, :, -1:].size()), std=1e-4*torch.ones(X_tuned[:, :, -1:].size())))
            X = X_tuned

        if self.transposition:
            transp = torch.randint(-self.transposition, self.transposition+1, (1, ))
            X_trans = torch.roll(X, (transp.item()*3, ), -1)
            y_trans = torch.roll(y, (transp.item(), ), -1)
            if transp>0:
                X_trans[:, :, :(3*transp)] = torch.abs(torch.normal(mean=torch.zeros(X_trans[:, :, :(3*transp)].size()), std=1e-4*torch.ones(X_trans[:, :, :(3*transp)].size())))
                y_trans[:, :, :transp] = torch.zeros(y_trans[:, :, :transp].size())
            elif transp<0:
                X_trans[:, :, (3*transp):] = torch.abs(torch.normal(mean=torch.zeros(X_trans[:, :, (3*transp):].size()), std=1e-4*torch.ones(X_trans[:, :, (3*transp):].size())))
                y_trans[:, :, transp:] = torch.zeros(y_trans[:, :, transp:].size())
            if y_trans.size(-1)==12:
                y_trans = torch.roll(y, (transp.item(), ), -1)
            X = X_trans
            y = y_trans

        return X, y


class dataset_context_segm(torch.utils.data.Dataset):
    """
    Dataset class to be used with DataLoader object. Generates HCQT segments with
    context. Note that X (HCQT input) includes the context frames but y (pitch
    (class) target) only refers to the center frames to be predicted.

    Args:
    inputs:         Tensor of HCQT input for one audio file
    targets:        Tensor of pitch (class) targets for the same audio file
    parameters:     Dictionary of parameters with:
    - 'context':        Total number of context frames +1 (=number of frames with seglenth=1)
    - 'seglength':      Length of the HCQT segment in frames (without context frames)
    - 'stride':         Hopsize for jumping to the start frame of the next segment
    - 'compression':    Gamma parameter for log compression of HCQT input
    - 'aug:transpsemitones':  Data augmentation with transposition (# semitones)
    - 'aug:scalingfactor':    Data augmentation with time scaling (factor)
    - 'aug:randomeq':         Data augmentation with random frequency equalization (amount)
    - 'aug:noisestd':         Data augmentation with random Gaussian noise (standard dev.)
    - 'aug:tuning':           Data augmentation with random tuning shift (+/- 1/3 semitone)
    """
    def __init__(self, inputs, targets, params):
        # Initialization
        #torch.initial_seed()
        self.inputs = inputs
        self.targets = targets
        self.context = params['context']
        self.seglength = params['seglength']
        self.stride = params['stride']
        self.compression = params['compression']
        self.transposition = None
        self.scalingfactor = None
        self.randomeq = None
        self.noisestd = None
        self.tuning = None
        if 'aug:transpsemitones' in params:
            self.transposition = params['aug:transpsemitones']
        if 'aug:scalingfactor' in params:
            self.scalingfactor = params['aug:scalingfactor']
        if 'aug:randomeq' in params:
            self.randomeq = params['aug:randomeq']
        if 'aug:noisestd' in params:
            self.noisestd = params['aug:noisestd']
        if 'aug:tuning' in params:
            self.tuning = params['aug:tuning']

    def __len__(self):
        # Denotes the total number of samples
        return (self.inputs.size()[1]-self.context-self.seglength+self.stride)//self.stride

    def __getitem__(self, index):
        # Generates one sample of data
        # jump to segment index*hopsize
        index *= self.stride
        # shift index by half context
        half_context = self.context//2
        index += half_context
        # get length of a segment
        seglength = self.seglength
        # Load data and get label
        X = self.inputs[:, (index-half_context):(index+seglength+half_context), :].type(torch.FloatTensor)
        y = torch.unsqueeze(torch.unsqueeze(self.targets[index:(index+seglength), :], 0), 1).type(torch.FloatTensor)

        if self.scalingfactor:
            scalefac = 1/self.scalingfactor + 2*torch.rand(1)*(1-1/self.scalingfactor)
            new_seglength = int(scalefac*self.seglength)
            # scale_transf =  transforms.Resize((X.size(2), new_seglength))
            # X_nocont = X[:, half_context:-half_context, :].transpose(1, 2)
            # X_scaled = scale_transf(X_nocont).transpose(1, 2)
            X_nocont = X[:, half_context:-half_context, :].transpose(1, 2)
            inputarray = X_nocont.numpy()
            xvec = np.array(range(inputarray.shape[2]))
            xnew = np.linspace(xvec.min(), xvec.max(), new_seglength)
            # apply the interpolation to each column
            f = interp1d(xvec, inputarray, axis=2, kind='linear')
            inputarr_scaled = f(xnew).astype('double')
            X_scaled = torch.from_numpy(inputarr_scaled).transpose(1, 2)
            X_scaled_context = torch.cat((X[:, :half_context, :], X_scaled, X[:, -half_context:, :]), dim=1)
            X = X_scaled_context.type(torch.FloatTensor)

        if self.randomeq:
            minval = -1
            while minval<0:
                randomAlpha = torch.randint(1, self.randomeq+1, (1,))
                randomBeta = torch.randint(0, 216, (1,))
                # filtvec = ((1 - (2e-6*randomAlpha*(torch.arange(216)-randomBeta)**2)).unsqueeze(0).unsqueeze(0))
                filtmat = torch.zeros((X.size(0), 1, X.size(2)))
                for nharm in range(filtmat.size(0)):
                    if nharm==0:
                        offset = int(-3*12)
                    else:
                        offset = int(3*12*(np.log2(nharm)))
                    randomBetaHarm = randomBeta - offset
                    currfiltvec = ((1 - (2e-6*randomAlpha*(torch.arange(216)-randomBetaHarm)**2)).unsqueeze(0).unsqueeze(0))
                    filtmat[nharm, :, :] = currfiltvec
                minval = torch.min(filtmat)
            X_filt = filtmat*X
            X = X_filt

        if self.noisestd:
            X += torch.normal(mean=torch.zeros(X.size()), std=self.noisestd*torch.ones(X.size()))
            X_noise = torch.abs(X)
            X = X_noise
            # X_pos = (X>0).type('torch.FloatTensor')

        if self.compression is not None:
            X = np.log(1+self.compression*X)

        if self.tuning:
            tuneshift = torch.randint(-2, 3, (1, )).item()
            tuneshift /= 2.
            X_tuned = X
            if tuneshift==0.5:
                # +0.5:
                X_tuned[:, :, 1:] = (X[:, :, :-1] + X[:, :, 1:])/2
            elif tuneshift==-0.5:
                # -0.5
                X_tuned[:, :, :-1] = (X[:, :, :-1] + X[:, :, 1:])/2
            else:
                X_tuned = torch.roll(X, (int(tuneshift), ), -1)
            if tuneshift>0:
                X_tuned[:, :, :1] = torch.abs(torch.normal(mean=torch.zeros(X_tuned[:, :, :1].size()), std=1e-4*torch.ones(X_tuned[:, :, :1].size())))
            elif tuneshift<0:
                X_tuned[:, :, -1:] = torch.abs(torch.normal(mean=torch.zeros(X_tuned[:, :, -1:].size()), std=1e-4*torch.ones(X_tuned[:, :, -1:].size())))
            X = X_tuned

        if self.transposition:
            transp = torch.randint(-self.transposition, self.transposition+1, (1, ))
            X_trans = torch.roll(X, (transp.item()*3, ), -1)
            y_trans = torch.roll(y, (transp.item(), ), -1)
            if transp>0:
                X_trans[:, :, :(3*transp)] = torch.abs(torch.normal(mean=torch.zeros(X_trans[:, :, :(3*transp)].size()), std=1e-4*torch.ones(X_trans[:, :, :(3*transp)].size())))
                y_trans[:, :, :transp] = torch.zeros(y_trans[:, :, :transp].size())
            elif transp<0:
                X_trans[:, :, (3*transp):] = torch.abs(torch.normal(mean=torch.zeros(X_trans[:, :, (3*transp):].size()), std=1e-4*torch.ones(X_trans[:, :, (3*transp):].size())))
                y_trans[:, :, transp:] = torch.zeros(y_trans[:, :, transp:].size())
            if y_trans.size(-1)==12:
                y_trans = torch.roll(y, (transp.item(), ), -1)
            X = X_trans
            y = y_trans

        return X, y


class dataset_context_segm_pitch(torch.utils.data.Dataset):
    """
    Dataset class to be used with DataLoader object. Generates HCQT segments with
    context. Note that X (HCQT input) includes the context frames but y (pitch
    target) only refers to the center frames to be predicted.

    Args:
    inputs:         Tensor of HCQT input for one audio file
    targets:        Tensor of pitch (class) targets for the same audio file
    parameters:     Dictionary of parameters with:
    - 'context':        Total number of context frames +1 (=number of frames with seglenth=1)
    - 'seglength':      Length of the HCQT segment in frames (without context frames)
    - 'stride':         Hopsize for jumping to the start frame of the next segment
    - 'compression':    Gamma parameter for log compression of HCQT input
    """
    def __init__(self, inputs, targets, params):
        # Initialization
        self.inputs = inputs
        self.targets = targets
        self.context = params['context']
        self.seglength = params['seglength']
        self.stride = params['stride']
        self.compression = params['compression']

    def __len__(self):
        # Denotes the total number of samples
        return (self.inputs.size()[1]-self.context-self.seglength+self.stride)//self.stride

    def __getitem__(self, index):
        # Generates one sample of data
        # jump to segment index*hopsize
        index *= self.stride
        # shift index by half context
        half_context = self.context//2
        index += half_context
        # get length of a segment
        seglength = self.seglength
        # Load data and get label
        X = self.inputs[:, (index-half_context):(index+seglength+half_context), :].type(torch.FloatTensor)
        if self.compression is not None:
            X = np.log(1+self.compression*X)
        y = torch.unsqueeze(torch.unsqueeze(self.targets[index:(index+seglength), 24:96], 0), 1).type(torch.FloatTensor)

        return X, y


class dataset_context_segm_widetarget(torch.utils.data.Dataset):
    """
    Dataset class to be used with DataLoader object. Generates HCQT segments with
    context. Note that X (HCQT input) includes the context frames but y (pitch
    target) only refers to the center frames to be predicted.

    Args:
    inputs:         Tensor of HCQT input for one audio file
    targets:        Tensor of pitch (class) targets for the same audio file
    parameters:     Dictionary of parameters with:
    - 'context':        Total number of context frames +1 (=number of frames with seglenth=1)
    - 'seglength':      Length of the HCQT segment CORRESPONDING TO THE TARGET
                            in frames (without context frames)
    - 'stride':         Hopsize for jumping to the start frame of the next segment
    - 'compression':    Gamma parameter for log compression of HCQT input
    """
    def __init__(self, inputs, targets, params):
        # Initialization
        self.inputs = inputs
        self.targets = targets
        self.context = params['context']
        self.seglength = params['seglength']
        self.stride = params['stride']
        self.compression = params['compression']

    def __len__(self):
        # Denotes the total number of samples
        return (self.inputs.size()[1]-self.context-self.seglength+self.stride)//self.stride

    def __getitem__(self, index):
        # Generates one sample of data
        # jump to segment index*hopsize
        segl_hcqt = 500
        index *= self.stride
        # shift index by half context
        half_context = self.context//2
        index += half_context
        # get length of a segment
        seglength = self.seglength
        # Compute start of HCQT patch
        index_hcqt = index + seglength//2 - segl_hcqt//2
        # Load data and get label
        X = self.inputs[:, (index_hcqt-half_context):(index_hcqt+segl_hcqt+half_context), :].type(torch.FloatTensor)
        if self.compression is not None:
            X = np.log(1+self.compression*X)
        y = torch.unsqueeze(torch.unsqueeze(self.targets[index:(index+seglength), :], 0), 1).type(torch.FloatTensor)

        return X, y


class dataset_context_measuresegm(torch.utils.data.Dataset):
    """
    Dataset class to be used with DataLoader object. Generates HCQT segments with
    context using measure positions (given) as segment boundaries. Note that X
    (HCQT input) includes the context frames but y (pitch (class) target) only
    refers to the center frames to be predicted.

    Args:
    inputs:         Tensor of HCQT input for one audio file
    targets:        Tensor of pitch (class) targets for the same audio file
    parameters:     Dictionary of parameters with:
    - 'context':        Total number of context frames +1 (=number of frames with seglenth=1)
    - 'seglength':      Length of the HCQT segment in measures(!) (without context frames)
    - 'stride':         Hopsize for jumping to the start frame of the next segment in measures(!)
    - 'compression':    Gamma parameter for log compression of HCQT input
    """
    def __init__(self, inputs, targets, measures, params):
        # Initialization
        torch.initial_seed()
        self.inputs = inputs
        self.targets = targets
        self.measures = measures
        self.context = params['context']
        self.seglength = params['seglength']
        self.stride = params['stride']
        self.compression = params['compression']

    def __len__(self):
        # Denotes the total number of samples
        return (self.measures.size()[0]-self.seglength-1)//self.stride  # skip last measure to guarantee enough context

    def __getitem__(self, index):
        # Generates one sample of data
        # jump to (measure) segment index*stride
        index *= self.stride
        # get start and end frame index
        start_frame = int(self.measures[index])
        end_frame = int(self.measures[index+self.seglength])
        # get half context
        half_context = self.context//2
        # get length of a segment
        # seglength_frames = end_frame-start_frame
        # Load data and get label
        X = self.inputs[:, (start_frame-half_context):(end_frame+half_context), :].type(torch.FloatTensor)
        if self.compression is not None:
            X = np.log(1+self.compression*X)
        y = torch.unsqueeze(torch.unsqueeze(self.targets[start_frame:end_frame, :], 0), 1).type(torch.FloatTensor)

        return X, y
