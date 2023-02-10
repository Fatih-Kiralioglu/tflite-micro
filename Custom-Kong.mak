#***********************************************************************************************************************
#
#	Custom MAKE variables for Logitech Bolide MicPod
#
#	Copyright (C) 2016-2022 - Logitech - Author: Andrea Santilli, Moussa Nasir, Miguel Blanco
#
#***********************************************************************************************************************

export CUSTOM = Logitech/Bolide/MicPod

#-----------------------------------------------------------------------------------------------------------------------

export ASDSP_HOME = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))../../../../..

export ASDSP_DEPS = $(ASDSP_HOME)/ASDSP/Make/$(CUSTOM)/Makefile $(ASDSP_HOME)/ASDSP/Make/$(CUSTOM)/Custom.mak

#-----------------------------------------------------------------------------------------------------------------------
#	Default ARCH definition
#-----------------------------------------------------------------------------------------------------------------------

ifndef ARCH
    ifeq ($(DESKTOP_SESSION), ubuntu)
    	ARCH = gnu_x64 
    endif
endif

#-----------------------------------------------------------------------------------------------------------------------
#	ARM Linux
#-----------------------------------------------------------------------------------------------------------------------

ifneq ($(findstring arm-, $(ARCH)),)

	ifneq ($(findstring arm-none-eabi, $(ARCH)),)				# XILINX ARM CORTEX (Parallel VM)
		#ARM_ARCH = ~/Xilinx/SDK/2016.4/gnu/aarch32/lin/gcc-arm-none-eabi/bin/arm-none-eabi
		ARM_ARCH = arm-xilinx-linux-gnueabi
		#ARM_ARCH = ~/Xilinx/SDK/2018.3/gnu/aarch32/lin/gcc-arm-none-eabi/bin/arm-none-eabi
		#ARM_ARCH = ~/Xilinx/SDK/2018.3/gnu/armr5/lin/gcc-arm-none-eabi/bin/armr5-none-eabi
		SKIP_TEST = 1
	endif
	ifneq ($(findstring arm-linux-gnueabihf, $(ARCH)),)			# for profiling purpose on Linux armhf
		ARM_ARCH=$(ARCH)
	endif
	CODE_GEN = -mfloat-abi=hard  #--specs=nosys.specs #--specs=rdimon.specs #-mfloat-abi=hard # -specs=rdimon.specs -marm -mthumb-interwork -Os

	export CC_OPTIONS = -fmessage-length=0 -ffast-math -funroll-loops -marm -march=armv7-a -mfpu=neon -fPIC -DPIC $(CODE_GEN)
	export LD_OPTIONS = -marm -march=armv7-a -mfpu=neon $(CODE_GEN)
endif

export CC = $(ARM_ARCH)-gcc
export AR = $(ARM_ARCH)-ar
export LD = $(ARM_ARCH)-g++
export CPP = $(ARM_ARCH)-g++

export CXX = $(ARM_ARCH)-g++
export TOOLCHAIN = $(ARM_ARCH)-gcc
export AR_TOOL = $(ARM_ARCH)-ar
export CC_TOOL = $(ARM_ARCH)-gcc
export CXX_TOOL = $(ARM_ARCH)-g++

#-----------------------------------------------------------------------------------------------------------------------
#	ASDSP Make variables
#-----------------------------------------------------------------------------------------------------------------------

export ASDSP_EXT_DEFINE =

export PFA_EXT_DEFINE = \
-D_PFA_FFT_8x2N_=1 \
-D_PFA_FFT_ZEROPAD_=1 \
-D_PFA_FFT_STD_NEW_=0 \
-D_PFA_FFT_SIZE_512_=1 \
-D_PFA_FFT_SIZE_128_=1 \
-D_CORR_IFFT_SIZE_128_=1 \
-DPFA_FFT_SIZE=64

#-----------------------------------------------------------------------------------------------------------------------
#	Generic MCHP settings
#-----------------------------------------------------------------------------------------------------------------------
export MCHP_EXT_DEFINE = \
-D_MICPOD_=1 \
-D_MCHP_=1 \
-DSAMPLE_FREQ=32000 \
-DFRAME_TIME=8 \
-DSUB_FRAME_TIME=2 \
-D_STATIC_=1

#-----------------------------------------------------------------------------------------------------------------------
#	AEC settings
#-----------------------------------------------------------------------------------------------------------------------
export MCHP_EXT_DEFINE += \
-DAEC_NUM_TAPS=15 \
-DAEC_NUM_TAPS_MIN=9 \
-DRESCUE_NUM_TAPS=3 \

#-----------------------------------------------------------------------------------------------------------------------
#	Beamformer settings
#-----------------------------------------------------------------------------------------------------------------------
# -D_SDMVDR_: square array differential MVDR
# -DDMVDR_MIC_SPACING: 32 mm spacing
# -DBEAM_HIGH_BAND: up to 6250 Hz
# -DBEAM_NUM_PARTS: up to 6250 Hz
# -DMPSR_HIGH_BAND: up to 3 kHz
# -DN_BEAMS: forced to 8 beams
export MCHP_EXT_DEFINE += \
-D_SDMVDR_=1 \
-DDMVDR_MIC_SPACING=0.032F \
-DBEAM_HIGH_BAND=PART_FREQ_8 \
-DBEAM_NUM_PARTS=9 \
-DMPSR_HIGH_BAND=96

#-----------------------------------------------------------------------------------------------------------------------
#	Postfilter settings
#-----------------------------------------------------------------------------------------------------------------------
# -D_SINGLE_TALK_DETECT_: 2 allows to force TalkState = SINGLE_TALK
# -D_DNN_MONITOR_: resets the DNN when deviating from DSP noise estimation. May be removed as set to 1 in Settings.h
# -D_SMALL_DNN_: no default value apparently, so set to false in Custom
# -DWARP_PSD_FILTER_NUM: number of Bark bands
# -DWARP_LOG_ENERGY_NUM_LAGS: number of lags for Bark spectrum
export MCHP_EXT_DEFINE += \
-D_SINGLE_TALK_DETECT_=2 \
-D_DNN_MONITOR_=1 \
-D_SMALL_DNN_=1 \
-DWARP_PSD_FILTER_NUM=48 \
-DWARP_LOG_ENERGY_NUM_LAGS=6

#-----------------------------------------------------------------------------------------------------------------------
#	Dereverb settings
#-----------------------------------------------------------------------------------------------------------------------
export MCHP_EXT_DEFINE += -D_DEREVERB_=0

#-----------------------------------------------------------------------------------------------------------------------
#	Hardware-dependent MCHP settings
#-----------------------------------------------------------------------------------------------------------------------
ifneq ($(strip $(findstring gnu, $(ARCH))	\
			   $(findstring arm-, $(ARCH))	\
			   $(findstring aarch64-linux-gnu, $(ARCH))),)
	export PFA_EXT_DEFINE += -D_FFT_NE10_=1
	export MCHP_EXT_DEFINE += -D_TUNING_=0
endif

ifneq ($(findstring arm-, $(ARCH)),)
	LIB_SUB_DIR = $(ARCH)
endif

ifneq ($(findstring _FFT_NE10_=1, $(PFA_EXT_DEFINE)),)
	export FFT_NE10_ROOT = $(ASDSP_HOME)
	export FFT_NE10_LIB_DIR = $(FFT_NE10_ROOT)/fft_ne10/$(LIB_SUB_DIR)
endif


#***********************************************************************************************************************
#
#	Copyright (C) 2016-2022 - Logitech
#
#***********************************************************************************************************************