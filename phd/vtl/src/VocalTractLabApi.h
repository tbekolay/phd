// ****************************************************************************
// This file defines the entry point for the DLL application, and the functions
// defined here are C-compatible so that they can be used with the 
// MATLAB shared library interface.
// ****************************************************************************

// Make an extern "C" section so that the functions can be accessed from Matlab

#ifdef __cplusplus
extern "C"{ /* start extern "C" */
#endif

// Definition for function export, if the file is compiled as part of a dll.

#ifdef WIN32
  #ifdef _USRDLL
    #define C_EXPORT __declspec(dllexport)
  #else
    #define C_EXPORT
  #endif        // DLL
#else
  #define C_EXPORT
#endif  // WIN32

// ****************************************************************************
// The exported C-compatible functions.
// IMPORTANT: 
// All the functions defined below must be named in the VocalTractLabApi.def 
// file in the project folder, so that they are usable from MATLAB !!!
// ****************************************************************************

// ****************************************************************************
// Init. the synthesis with the given speaker file name, e.g. "JD2.speaker".
// This function MUST be called before any other function of this API.
// Returns 0 in case of success, otherwise an error code > 0.
// ****************************************************************************

C_EXPORT int vtlInitialize(const char *speakerFileName);

// ****************************************************************************
// Returns the version of this API as a string that contains the compile data.
// Reserve at least 32 chars for the string.
// ****************************************************************************

C_EXPORT void vtlGetVersion(char *version);

// ****************************************************************************
// Returns a couple of constants:
// o The audio sampling rate of the synthesized signal.
// o The number of supraglottal tube sections.
// o The number of vocal tract model parameters.
// o The number of glottis model parameters.
// ****************************************************************************

C_EXPORT void vtlGetConstants(int *audioSamplingRate, int *numTubeSections,
  int *numVocalTractParams, int *numGlottisParams);

// ****************************************************************************
// Returns for each vocal tract parameter the minimum value, the maximum value,
// and the neutral value. Each vector passed to this function must have at 
// least as many elements as the number of vocal tract model parameters.
// The "names" string receives the abbreviated names of the parameters separated
// by spaces. This string should have at least 10*numParams elements.
// ****************************************************************************

C_EXPORT void vtlGetTractParamInfo(char *names, double *paramMin, double *paramMax, double *paramNeutral);

// ****************************************************************************
// Returns for each glottis model parameter the minimum value, the maximum value,
// and the neutral value. Each vector passed to this function must have at 
// least as many elements as the number of glottis model parameters.
// The "names" string receives the abbreviated names of the parameters separated
// by spaces. This string should have at least 10*numParams elements.
// ****************************************************************************

C_EXPORT void vtlGetGlottisParamInfo(char *names, double *paramMin, double *paramMax, double *paramNeutral);

// ****************************************************************************
// Returns the vocal tract parameters for the given shape as defined in the
// speaker file.
// The vector passed to this function must have at least as many elements as 
// the number of vocal tract model parameters.
// Returns 0 in the case of success, or 1 if the shape is not defined.
// ****************************************************************************

C_EXPORT int vtlGetTractParams(char *shapeName, double *param);

// ****************************************************************************
// Synthesize speech with a given sequence of vocal tract model states and 
// glottis model states, and return the corresponding sequence of tube
// area functions and the audio signal.
// Parameters (in/out):
// o tractParams (in): Is a concatenation of vocal tract parameter vectors
//     with the total length of (numVocalTractParams*numFrames) elements.
// o glottisParams (in): Is a concatenation of glottis parameter vectors
//     with the total length of (numGlottisParams*numFrames) elements.
// o tubeAreas (out): Is a concatenation of vocal tract area functions that
//     result from the vocal tract computations. Reserve (numTubeSections*numFrames)
//     elements for this vector!
// o numFrames (in): Number of successive states of the glottis and vocal tract
//     that are going to be concatenated.
// o frameRate_Hz (in): The number of frames (states) per second.
// o audio (out): The resulting audio signal with sample values in the range 
//     [-1, +1] and with the sampling rate audioSamplingRate. Reserve enough
//     elements for the samples at the given frame rate, sampling rate, and
//     number of frames!
// o numAudioSamples (out): The number of audio samples written into the
//     audio array.
//
// Returns 0 in case of success, otherwise an error code > 0.
// ****************************************************************************

C_EXPORT int vtlSynthBlock(double *tractParams, double *glottisParams, double *tubeAreas,
  int numFrames, double frameRate_Hz, double *audio, int *numAudioSamples);

// ****************************************************************************
// Test function for this API.
// Audio should contain at least 44100 double values.
// Run this function directly WITHOUT calling vtlInitialize() and vtlClose() !
// ****************************************************************************

C_EXPORT void vtlApiTest(const char *speakerFileName, double *audio, int *numSamples);

// ****************************************************************************
/// This function directly converts a gestural score for a given speaker into
/// an audio file and a corresponding file that contains the sequence of vocal
/// tract area functions.
// Run this function directly WITHOUT calling vtlInitialize() and vtlClose() !
/// Parameters:
/// o speakerFileName: The name of the speaker file to use for the synthesis.
///   The synthesizer will use the glottis model selected in the speaker file!
/// o gestureFileName: Name of the gestural score file to synthesize.
/// o wavFileName: Name of the audio file to where the resulting speech signal
///   will be written.
/// o feedbackFileName: Name of a text file to which the sequence of area 
///   functions and other feedback data corresponding to the gestural score 
///   will be written. You can pass NULL (0)
///   for this parameter if you are not interested in this data.
///
/// The return value is zero if successful, and otherwise an error code >= 1.
/// Error codes:
/// 1: Loading the speaker file failed.
/// 2: Loading the gestural score file failed.
/// 3: Invalid wav-file name.
/// 4: The wav-file could not be saved.
// ****************************************************************************

C_EXPORT int vtlGesToWav(const char *speakerFileName, const char *gestureFileName,
  const char *wavFileName, const char *feedbackFileName);

// ****************************************************************************
// Clean up the memory and shut down the synthesizer.
// ****************************************************************************

C_EXPORT void vtlClose();

// ****************************************************************************

#ifdef __cplusplus
} /* end extern "C" */
#endif
