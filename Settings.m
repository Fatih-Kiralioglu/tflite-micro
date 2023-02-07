Program           = 'Test_DNN';
Platform          = 'Mac-Float';
MCHP_VER          = '01.13Jb';
MCHP_DATE         = '2022/11/23';

FRAME_TIME = 8;
WARP_PSD_FILTER_NUM = 48;
WARP_LOG_ENERGY_NUM_LAGS = 6;

DNN_Model = 'cs_37';

InputList = ...
{
    'DNN_INPUT(0, WarpLogEnergy, 288)'
};

OutputList = ...
{
    'DNN_OUTPUT(0, SpeechProbability, 0, 0)'
    'DNN_OUTPUT(1, VoiceNoiseRatio, 1, 0)'
    'DNN_OUTPUT(2, NonSpeechRatio, 2, 0, 48)'
};

LayerList = ...
{
    'NORM(0, 0, 288)'
    'CONV2D(0, 2, 6, 48, 32, 3, 48, 1, 1, Relu)'
    'CONV2D(0, 3, 4, 32, 32, 3, 32, 1, 1, Relu)'
    'GRU(0, 6, 64, 48, Tanh, Sigmoid)'
    'GRU(0, 8, 48, 32, Tanh, Sigmoid)'
    'DENSE(0, 10, 32, 48, Relu)'
    'SPLIT(1, 0, 48)'
    'SPLIT(2, 0, 48)'
    'DENSE(0, 12, 48, 1, Sigmoid)'
    'DENSE(1, 0, 48, 1, Sigmoid)'
    'DENSE(2, 0, 48, 48, Sigmoid)'
};

NumberOfOperations = 61156;

