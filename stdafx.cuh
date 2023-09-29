#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include <errno.h>
#include <float.h>
#include <math.h>
#include <memory.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#ifdef LINUX
#include <sys/resource.h>
#endif

#ifdef LINUX
#include <unistd.h>
#endif

#include "showdata.cuh"
#include "number.cuh"

void afxsigp(void);

extern volatile int break_loop;

void exit_function();