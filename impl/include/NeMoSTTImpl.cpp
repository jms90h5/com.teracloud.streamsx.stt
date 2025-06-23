#include "NeMoSTTImpl.hpp"
#include <iostream>
#include <sstream>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace com::teracloud::streams::stt {

bool NeMoSTTImpl::pythonInitialized_ = false;
int NeMoSTTImpl::instanceCount_ = 0;

NeMoSTTImpl::NeMoSTTImpl() 
    : pModule_(nullptr), pModel_(nullptr), pTranscribeFunc_(nullptr), initialized_(false) {
    
    // Initialize Python if not already done
    if (!pythonInitialized_) {
        Py_Initialize();
        
        // Initialize NumPy
        _import_array();
        
        // Add current directory to Python path
        PyObject* sysPath = PySys_GetObject("path");
        PyObject* currentDir = PyUnicode_FromString(".");
        PyList_Append(sysPath, currentDir);
        Py_DECREF(currentDir);
        
        pythonInitialized_ = true;
    }
    
    instanceCount_++;
}

NeMoSTTImpl::~NeMoSTTImpl() {
    // Clean up Python objects
    Py_XDECREF(pTranscribeFunc_);
    Py_XDECREF(pModel_);
    Py_XDECREF(pModule_);
    
    instanceCount_--;
    
    // Finalize Python if this was the last instance
    if (instanceCount_ == 0 && pythonInitialized_) {
        Py_Finalize();
        pythonInitialized_ = false;
    }
}

bool NeMoSTTImpl::initialize(const std::string& modelPath) {
    modelPath_ = modelPath;
    
    // Create Python code to load NeMo model
    std::string pythonCode = R"(
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nemo.collections.asr as nemo_asr
import torch
import librosa

# Global model instance
model = None

def initialize_model(model_path):
    global model
    try:
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def transcribe_audio(audio_data, sample_rate):
    global model
    try:
        # Convert to numpy array if needed
        audio_np = np.array(audio_data, dtype=np.float32)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
        
        # Save to temporary file (NeMo expects file paths)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, audio_np, 16000)
            tmp_path = tmp_file.name
        
        # Transcribe
        transcriptions = model.transcribe([tmp_path])
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Extract text from result
        if isinstance(transcriptions[0], str):
            return transcriptions[0]
        else:
            return transcriptions[0][0]
            
    except Exception as e:
        return f"Error: {str(e)}"
)";
    
    // Execute the Python code
    PyObject* pGlobals = PyDict_New();
    PyObject* pLocals = PyDict_New();
    
    PyObject* pCode = PyRun_String(pythonCode.c_str(), Py_file_input, pGlobals, pLocals);
    if (!pCode) {
        checkPythonError();
        Py_DECREF(pGlobals);
        Py_DECREF(pLocals);
        return false;
    }
    Py_DECREF(pCode);
    
    // Get references to the functions
    PyObject* pInitFunc = PyDict_GetItemString(pLocals, "initialize_model");
    PyObject* pTransFunc = PyDict_GetItemString(pLocals, "transcribe_audio");
    
    if (!pInitFunc || !pTransFunc) {
        setError("Failed to get Python functions");
        Py_DECREF(pGlobals);
        Py_DECREF(pLocals);
        return false;
    }
    
    // Call initialize_model
    PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(modelPath.c_str()));
    PyObject* pResult = PyObject_CallObject(pInitFunc, pArgs);
    Py_DECREF(pArgs);
    
    if (!pResult) {
        checkPythonError();
        Py_DECREF(pGlobals);
        Py_DECREF(pLocals);
        return false;
    }
    
    bool success = PyObject_IsTrue(pResult);
    Py_DECREF(pResult);
    
    if (success) {
        // Keep references to the functions
        pTranscribeFunc_ = pTransFunc;
        Py_INCREF(pTranscribeFunc_);
        pModule_ = pLocals;  // Keep the locals dict
        initialized_ = true;
    } else {
        Py_DECREF(pGlobals);
        Py_DECREF(pLocals);
    }
    
    return success;
}

std::string NeMoSTTImpl::transcribe(const std::vector<float>& audioData, int sampleRate) {
    if (!initialized_) {
        return "Error: Model not initialized";
    }
    
    // Convert audio data to NumPy array
    npy_intp dims[1] = {static_cast<npy_intp>(audioData.size())};
    PyObject* pArray = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, 
                                                  const_cast<float*>(audioData.data()));
    
    // Call transcribe_audio function
    PyObject* pArgs = PyTuple_Pack(2, pArray, PyLong_FromLong(sampleRate));
    PyObject* pResult = PyObject_CallObject(pTranscribeFunc_, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pArray);
    
    if (!pResult) {
        checkPythonError();
        return "Error: Transcription failed";
    }
    
    // Extract string result
    std::string transcription;
    if (PyUnicode_Check(pResult)) {
        const char* str = PyUnicode_AsUTF8(pResult);
        if (str) {
            transcription = str;
        }
    }
    
    Py_DECREF(pResult);
    return transcription;
}

void NeMoSTTImpl::setError(const std::string& error) {
    lastError_ = error;
    std::cerr << "NeMoSTT Error: " << error << std::endl;
}

bool NeMoSTTImpl::checkPythonError() {
    if (PyErr_Occurred()) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        
        if (pvalue) {
            PyObject* pStr = PyObject_Str(pvalue);
            if (pStr) {
                const char* errStr = PyUnicode_AsUTF8(pStr);
                if (errStr) {
                    setError(errStr);
                }
                Py_DECREF(pStr);
            }
        }
        
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        
        PyErr_Clear();
        return true;
    }
    return false;
}

} // namespace com::teracloud::streams::stt