#include <jni.h>
#include "../jni.h"
#include "LlamaManager.h"

llama_context_params
LlamaManager::LlamaSession::LlamaContextParamsManager::getParams() {
  return llamaContextParams;
}

LlamaManager::LlamaSession::LlamaContextParamsManager::LlamaContextParamsManager(jobject javaContextParams, LlamaSession* session) : session(session) {

  JNIEnv* env = session->env;
  jclass javaParamsClass = env->GetObjectClass(javaContextParams);

  tensorSplitFloatArray = jni::getJFloatArray(env,
                                              javaParamsClass,
                                              javaContextParams,
                                              "tensorSplit");
  tensorSplit = tensorSplitFloatArray ? env->GetFloatArrayElements(tensorSplitFloatArray, nullptr) : nullptr;

  jfieldID callbackFieldId = env->GetFieldID( javaParamsClass, "progressCallback", "Ljava/util/function/Consumer;");
  if (callbackFieldId == nullptr) {
    throw jni::JNIException("field \"progressCallback\" must exist on Java LlamaContextParams class");
  }
  jobject jprogressCallback = env->GetObjectField(javaContextParams, callbackFieldId);

  ProgressContext* callbackContext;

  if (jprogressCallback) {
    callbackContext = new ProgressContext{
      env->NewGlobalRef(jprogressCallback)
    };
    env->DeleteLocalRef(jprogressCallback);
  } else {
    callbackContext = nullptr;
  }

  llamaContextParams = {
      jni::getUnsignedInt32(env, javaParamsClass, javaContextParams, "seed"),
      jni::getInt32(env, javaParamsClass, javaContextParams, "nCtx"),
      jni::getInt32(env, javaParamsClass, javaContextParams, "nBatch"),
      jni::getInt32(env, javaParamsClass, javaContextParams, "nGqa"),
      jni::getFloat(env, javaParamsClass, javaContextParams, "rmsNormEps"),
      jni::getInt32(env, javaParamsClass, javaContextParams, "nGpuLayers"),
      jni::getInt32(env, javaParamsClass, javaContextParams, "mainGpu"),
      tensorSplit,
      jni::getFloat(env, javaParamsClass, javaContextParams, "ropeFreqBase"),
      jni::getFloat(env, javaParamsClass, javaContextParams, "ropeFreqScale"),
      progressCallback,
      callbackContext,
      jni::getBool(env, javaParamsClass, javaContextParams, "lowVram"),
      jni::getBool(env, javaParamsClass, javaContextParams, "mulMatQ"),
      jni::getBool(env, javaParamsClass, javaContextParams, "f16Kv"),
      jni::getBool(env, javaParamsClass, javaContextParams, "logitsAll"),
      jni::getBool(env, javaParamsClass, javaContextParams, "vocabOnly"),
      jni::getBool(env, javaParamsClass, javaContextParams, "useMmap"),
      jni::getBool(env, javaParamsClass, javaContextParams, "useMlock"),
      jni::getBool(env, javaParamsClass, javaContextParams, "embedding")
  };
}

LlamaManager::LlamaSession::LlamaContextParamsManager::~LlamaContextParamsManager() {
  if (tensorSplit) {
    session->env->ReleaseFloatArrayElements(tensorSplitFloatArray,
                                   (jfloat*) tensorSplit,
                                   JNI_ABORT);
  }
}
