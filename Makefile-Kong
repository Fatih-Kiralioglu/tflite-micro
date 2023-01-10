#******************************************************************************
#
#	Top Level Makefile for Logitech Bolide MicPod
#
#	Copyright (C) 2003-2017 -- Logitech -- Author: Andrea Santilli
#
#******************************************************************************

include Custom-Kong.mak

#----------------------------------------------------------

ALL_ARCH = arm-xilinx-linux-gnueabi

#----------------------------------------------------------


#----------------------------------------------------------

ALL_MODULES = tflm

ALL_MODULE_RELEASE = MAKE_TARGET="hello_world_test" $(ALL_MODULES)

#----------------------------------------------------------

arm-none-eabi:
	make ARCH=arm-xilinx-linux-gnueabi TARGET_ARCH=arm-xilinx-linux-gnueabi $(ALL_MODULE_RELEASE)

#----------------------------------------------------------

tflm:
	make -f tensorflow/lite/micro/tools/make/Makefile TARGET_ARCH=arm-xilinx-linux-gnueabi			$(MAKE_TARGET)

#----------------------------------------------------------


#******************************************************************************
