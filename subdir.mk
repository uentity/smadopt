################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../alg.cpp \
../ga.cpp \
../ga_opt.cpp \
../ga_win.cpp \
../km_opt.cpp \
../kmeans.cpp \
../mnet.cpp \
../nn_addon.cpp \
../nn_except.cpp \
../nn_opt.cpp \
../nna_opt.cpp \
../objnet.cpp 

CPP_DEPS += \
./alg.d \
./ga.d \
./ga_opt.d \
./ga_win.d \
./km_opt.d \
./kmeans.d \
./mnet.d \
./nn_addon.d \
./nn_except.d \
./nn_opt.d \
./nna_opt.d \
./objnet.d 

OBJS += \
./alg.o \
./ga.o \
./ga_opt.o \
./ga_win.o \
./km_opt.o \
./kmeans.o \
./mnet.o \
./nn_addon.o \
./nn_except.o \
./nn_opt.o \
./nna_opt.o \
./objnet.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D_UNIX -D__EXPORTING -I"/mnt/D/MyProjects/hybrid_adapt/lin/utils" -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fno-rtti -fpermissive -fdiagnostics-show-location=once -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


