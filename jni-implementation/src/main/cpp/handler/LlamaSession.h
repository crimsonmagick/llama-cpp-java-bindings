#ifndef JNI_IMPLEMENTATION_LLAMASESSION_H
#define JNI_IMPLEMENTATION_LLAMASESSION_H

#include <jni.h>
#include "../exceptions/DynamicLibraryException.h"
#include "../jni.h"

class LlamaSession {
  public:
  LlamaSession(JNIEnv* env) : env(env) {}
  void backendInit(bool useNuma);

  private:
  JNIEnv* env;
  template<typename Func>
  auto withJniExceptions(Func&& func) -> decltype(func()) {
    using ReturnType = decltype(func());
    try {
      return func();
    } catch (const DynamicLibraryException &e) {
      jni::throwDLLException(env, e);
    } catch (const jni::JNIException &e) {
      jni::throwJNIException(env, e);
    } catch (const LlamaCppException &e) {
      jni::throwLlamaCppException(env, e);
    }
    return ReturnType();
  }
};


#endif //JNI_IMPLEMENTATION_LLAMASESSION_H