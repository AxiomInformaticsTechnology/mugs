#include "stdafx.cuh"

FILE * stddoc;

typedef void (*SignalHandlerPointer)(int);

SignalHandlerPointer previousHandlerSigInt;
SignalHandlerPointer previousHandlerSigFpe;
SignalHandlerPointer previousHandlerSigSegv;
SignalHandlerPointer previousHandlerAbort;
SignalHandlerPointer previousHandlerTerm;

void SigIntSignalHandler(int signal_code)
{    
	break_loop = 1;
}
void SigFpeSignalHandler(int signal_code)
{
	signal(SIGFPE, previousHandlerSigFpe);
	break_loop = 2;
	exit_function();
	raise(SIGFPE);
}
void SigSegvSignalHandler(int signal_code)
{
	signal(SIGSEGV, previousHandlerSigSegv);
	break_loop = 3;
	exit_function();
	raise(SIGSEGV);
}
void AbortSignalHandler(int signal_code)
{
	signal(SIGABRT, previousHandlerAbort);
	break_loop = 4;
	exit_function();
	raise(SIGABRT);
}
void TermSignalHandler(int signal_code)
{
	signal(SIGTERM, previousHandlerTerm);
	break_loop = 5;
	exit_function();
	raise(SIGTERM);
}

void afxsigp(void)
{
	previousHandlerTerm = signal(SIGTERM, TermSignalHandler);
	previousHandlerAbort = signal(SIGABRT, AbortSignalHandler);
	previousHandlerSigInt = signal(SIGINT, SigIntSignalHandler);
	previousHandlerSigFpe = signal(SIGFPE, SigFpeSignalHandler);
	previousHandlerSigSegv = signal(SIGSEGV, SigSegvSignalHandler);
	stddoc = 0;
}

void exit_function()
{
	printf("\nFinished\n");
	return;
}