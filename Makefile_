#******************************************************************************
#
#	Top Level Makefile for Logitech Bolide MicPod
#
#	Copyright (C) 2003-2017 -- Logitech -- Author: Andrea Santilli
#
#******************************************************************************

include Custom.mak

#----------------------------------------------------------

ALL_ARCH = arm-none-eabi

#----------------------------------------------------------


#----------------------------------------------------------

ALL_MODULES = tflm

ALL_MODULE_RELEASE = MAKE_TARGET="hello_world_test" $(ALL_MODULES)

#----------------------------------------------------------

arm-none-eabi:
	make ARCH=arm-none-eabi TARGET_ARCH=arm-none-eabi $(ALL_MODULE_RELEASE)

#----------------------------------------------------------

tflm:
	make -f tensorflow/lite/micro/tools/make/Makefile TARGET_ARCH=arm-none-eabi			$(MAKE_TARGET)

#----------------------------------------------------------


#******************************************************************************
