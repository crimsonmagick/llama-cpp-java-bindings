package net.jllama.api;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;
import net.jllama.api.exceptions.LlamaApiException;
import net.jllama.api.exceptions.MissingParameterException;
import net.jllama.core.LlamaCpp;
import net.jllama.core.LlamaLogLevel;
import net.jllama.core.LlamaModel;
import net.jllama.core.LlamaModelParams;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Llama implements Closeable {

  final private static Logger log = LogManager.getLogger(Llama.class);
  private static Llama singleton;

  private static Boolean initializedUseNuma;

  private boolean closed;

  public static Llama library() {
    return library(false);
  }
  public static synchronized Llama library(final boolean useNuma) {
    if (singleton != null) {
      if (useNuma != initializedUseNuma) {
        throw new LlamaApiException(String.format("Llama.CPP has already been initialized with useNuma=%s", initializedUseNuma));
      }
      return singleton;
    }
    LlamaCpp.loadLibrary();
    singleton = new Llama(useNuma);
    initializedUseNuma = useNuma;
    return singleton;
  }

  private static void logLllamaMessage(final LlamaLogLevel logLevel, final byte[] message) {
    final String messageText = new String(message, StandardCharsets.UTF_8);
    if (logLevel == LlamaLogLevel.INFO) {
      log.info(messageText);
    } else if (logLevel == LlamaLogLevel.WARN) {
      log.warn(messageText);
    } else {
      log.error(messageText);
    }
  }

  public class ModelConfigurer {

    public class ParametrizedLoader {

      private ParametrizedLoader() {
        llamaModelParams = new LlamaModelParams();
      }

      private ParametrizedLoader(final LlamaModelParams llamaModelParams) {
        this.llamaModelParams = llamaModelParams;
      }

      private final LlamaModelParams llamaModelParams;
      private String path;

      public Model load() {
        return loadModel(path, llamaModelParams);
      }

      public ParametrizedLoader path(final String path) {
        this.path = path;
        return this;
      }

      public ParametrizedLoader gpuLayerCount(final int gpuLayerCount) {
        llamaModelParams.setnGpuLayers(gpuLayerCount);
        return this;
      }

      public ParametrizedLoader mainGpuIndex(final int mainGpuIndex) {
        llamaModelParams.setMainGpu(mainGpuIndex);
        return this;
      }

      public ParametrizedLoader tensorSplit(final float[] tensorSplit) {
        llamaModelParams.setTensorSplit(tensorSplit);
        return this;
      }

      public ParametrizedLoader vocabOnly(final boolean vocabOnly) {
        llamaModelParams.setVocabOnly(vocabOnly);
        return this;
      }

      public ParametrizedLoader useMmap(final boolean useMmap) {
        llamaModelParams.setUseMmap(useMmap);
        return this;
      }

      public ParametrizedLoader useMlock(final boolean useMlock) {
        llamaModelParams.setUseMlock(useMlock);
        return this;
      }

    }

    public ParametrizedLoader withDefaults() {
      return new ParametrizedLoader(LlamaModel.llamaModelDefaultParams());
    }

    public ParametrizedLoader with() {
      return new ParametrizedLoader();
    }

  }

  private Llama(final boolean useNuma) {
    initializedUseNuma = useNuma;
    LlamaCpp.loadLibrary();
    LlamaCpp.llamaBackendInit(useNuma);
    LlamaCpp.llamaLogSet(Llama::logLllamaMessage);
  }

  public ModelConfigurer newModel() {
    return new ModelConfigurer();
  }

  private Model loadModel(final String path, final LlamaModelParams llamaModelParams) {
    if (path == null || path.isEmpty()) {
      throw new MissingParameterException("Model cannot be loaded. Path is a required parameter.");
    }
    // TODO add progress logging
    final LlamaModel llamaModel = LlamaCpp.loadModel(path.getBytes(StandardCharsets.UTF_8), llamaModelParams);
    return new Model(llamaModel);
  }

  @Override
  public synchronized void close() {
    // TODO add method for deregistering the logger (avoid a possible memory leak)
    if (!closed) {
      LlamaCpp.llamaBackendFree();
      LlamaCpp.closeLibrary();
      initializedUseNuma = null;
      singleton = null;
      closed = true;
    }
  }

}
